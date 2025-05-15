async function initParticleSystem() {
    if (!navigator.gpu) {
        document.body.innerHTML = "<h1 style='color:white;text-align:center;padding:2rem'>WebGPU not supported in this browser</h1>";
        return;
    }
    
    // Initialize WebGPU
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    
    // Canvas setup
    const canvas = document.getElementById("webgpu-canvas");
    const context = canvas.getContext("webgpu");
    
    // Handle canvas resize
    function resizeCanvas() {
        canvas.width = window.innerWidth * devicePixelRatio;
        canvas.height = window.innerHeight * devicePixelRatio;
    }
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ 
        device, 
        format, 
        alphaMode: "premultiplied" 
    });
    
    // Particle system settings
    const particleCount = 1024;
    const positions = new Float32Array(particleCount * 3);
    const velocities = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    
    // Initialize particles
    for (let i = 0; i < particleCount; i++) {
        // Positions: distribute randomly in a sphere
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        const r = 2.0 * Math.pow(Math.random(), 1/3); // Cubic root for uniform distribution
        
        positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);     // x
        positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta); // y
        positions[i * 3 + 2] = r * Math.cos(phi);                  // z
        
        // Velocities: small random initial values
        velocities[i * 3] = (Math.random() - 0.5) * 0.2;
        velocities[i * 3 + 1] = (Math.random() - 0.5) * 0.2;
        velocities[i * 3 + 2] = (Math.random() - 0.5) * 0.2;
        
        // Colors: based on position for visual interest
        colors[i * 3] = 0.5 + Math.sin(positions[i * 3]) * 0.5;     // r
        colors[i * 3 + 1] = 0.5 + Math.sin(positions[i * 3 + 1]) * 0.5; // g
        colors[i * 3 + 2] = 0.5 + Math.cos(positions[i * 3 + 2]) * 0.5; // b
    }
    
    // Create buffers
    const positionBuffer = device.createBuffer({
        size: positions.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(positionBuffer.getMappedRange()).set(positions);
    positionBuffer.unmap();
    
    const velocityBuffer = device.createBuffer({
        size: velocities.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(velocityBuffer.getMappedRange()).set(velocities);
    velocityBuffer.unmap();
    
    const colorBuffer = device.createBuffer({
        size: colors.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(colorBuffer.getMappedRange()).set(colors);
    colorBuffer.unmap();
    
    // Cylinder data: centerX, centerY, centerZ, radius, height (flattened array)
    const cylinderData = new Float32Array([
        0, 0, 0, 1.0, 2.0,    // Cylinder 1: center(0,0,0), radius 1.0, height 2.0
        2, 0, 2, 0.5, 1.5,    // Cylinder 2: center(2,0,2), radius 0.5, height 1.5
        -2, 0, -2, 0.75, 3.0  // Cylinder 3: center(-2,0,-2), radius 0.75, height 3.0
    ]);
    
    const cylinderBuffer = device.createBuffer({
        size: cylinderData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Float32Array(cylinderBuffer.getMappedRange()).set(cylinderData);
    cylinderBuffer.unmap();
    
    // Create uniform buffer for camera and simulation parameters
    // Get UI control elements
    const repulsionSlider = document.getElementById('repulsion-strength');
    const attractionSlider = document.getElementById('attraction-strength');
    
    const uniformData = new Float32Array([
        // Camera position (xyz) and aspect ratio
        0, 0, -10, canvas.width / canvas.height,
        // Time, dt, numCylinders, _pad1
        0, 0.016, 3, 0,
        // Force parameters and padding
        parseFloat(repulsionSlider.value),
        parseFloat(attractionSlider.value),
        0, // _pad2
        0  // _pad3 (additional padding to reach 48 bytes)
    ]);
    
    const uniformBuffer = device.createBuffer({
        size: uniformData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Float32Array(uniformBuffer.getMappedRange()).set(uniformData);
    uniformBuffer.unmap();
    
    // Load shader code
    const shaderCode = await fetch('particles.wgsl').then(response => {
        if (!response.ok) {
            throw new Error(`Failed to load shader: ${response.status} ${response.statusText}`);
        }
        return response.text();
    });
    
    // Create shader module
    const shaderModule = device.createShaderModule({
        code: shaderCode
    });
    
    // Create compute pipeline
    const computePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { 
            module: shaderModule, 
            entryPoint: 'computeMain' 
        }
    });
    
    // Create render pipeline
    const renderPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
            module: shaderModule,
            entryPoint: 'vertexMain',
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fragmentMain',
            targets: [{ format }]
        },
        primitive: {
            topology: 'point-list',
            cullMode: 'none'
        },
        multisample: {
            count: 1
        },
        depthStencil: {
            depthWriteEnabled: true,
            depthCompare: 'less',
            format: 'depth24plus'
        }
    });
    
    // Create depth texture
    let depthTexture = device.createTexture({
        size: [canvas.width, canvas.height],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
    
    // Create bind group for compute pipeline
    const computeBindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: positionBuffer }},
            { binding: 1, resource: { buffer: velocityBuffer }},
            { binding: 2, resource: { buffer: colorBuffer }},
            { binding: 3, resource: { buffer: cylinderBuffer }},
            { binding: 4, resource: { buffer: uniformBuffer }}
        ]
    });
    
    // Create bind group for render pipeline
    const renderBindGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: positionBuffer }},
            { binding: 2, resource: { buffer: colorBuffer }},
            { binding: 4, resource: { buffer: uniformBuffer }}
        ]
    });
    
    // Animation state
    let startTime = performance.now() / 1000;
    let lastFrameTime = startTime;
    
    // Camera animation parameters
    const cameraRadius = 10;
    const cameraHeight = 3;
    const cameraSpeed = 0.3;
    
    // Update UI value displays
    repulsionSlider.addEventListener('input', () => {
        document.getElementById('repulsion-value').textContent = repulsionSlider.value;
        device.queue.writeBuffer(
            uniformBuffer,
            32, // Offset to repulsionStrength
            new Float32Array([parseFloat(repulsionSlider.value)])
        );
    });
    
    attractionSlider.addEventListener('input', () => {
        document.getElementById('attraction-value').textContent = attractionSlider.value;
        device.queue.writeBuffer(
            uniformBuffer,
            36, // Offset to attractionStrength
            new Float32Array([parseFloat(attractionSlider.value)])
        );
    });

    // Animation loop
    function frame() {
        // Handle canvas resize if needed
        if (canvas.width !== window.innerWidth * devicePixelRatio || 
            canvas.height !== window.innerHeight * devicePixelRatio) {
            resizeCanvas();
            
            // Update aspect ratio
            device.queue.writeBuffer(
                uniformBuffer, 
                12, // Offset to aspectRatio (after cameraPos)
                new Float32Array([canvas.width / canvas.height])
            );
            
            // Recreate depth texture
            depthTexture.destroy();
            depthTexture = device.createTexture({
                size: [canvas.width, canvas.height],
                format: 'depth24plus',
                usage: GPUTextureUsage.RENDER_ATTACHMENT
            });
        }
        
        // Update time parameters
        const now = performance.now() / 1000;
        const dt = Math.min(0.033, now - lastFrameTime); // Cap at 30fps equivalent
        lastFrameTime = now;
        const time = now - startTime;
        
        // Animate camera position in a circular path
        const cameraX = cameraRadius * Math.sin(time * cameraSpeed);
        const cameraZ = cameraRadius * Math.cos(time * cameraSpeed);
        const cameraY = cameraHeight * Math.sin(time * cameraSpeed * 0.5);
        
        // Update uniform buffer with new camera position and time
        device.queue.writeBuffer(
            uniformBuffer,
            0, // Offset to start at beginning
            new Float32Array([
                cameraX, cameraY, cameraZ, canvas.width / canvas.height,
                time, dt, 3, 0 // time, dt, numCylinders, _pad
            ])
        );
        
        // Start command encoder
        const commandEncoder = device.createCommandEncoder();
        
        // Compute pass
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(computePipeline);
        computePass.setBindGroup(0, computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(particleCount / 64));
        computePass.end();
        
        // Render pass
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.0, g: 0.0, b: 0.08, a: 1.0 } // Dark blue background
            }],
            depthStencilAttachment: {
                view: depthTexture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store'
            }
        });
        
        renderPass.setPipeline(renderPipeline);
        renderPass.setBindGroup(0, renderBindGroup);
        renderPass.draw(particleCount);
        renderPass.end();
        
        // Submit and schedule next frame
        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    }
    
    // Start animation
    requestAnimationFrame(frame);
}

// Initialize everything when the page loads
window.addEventListener('load', initParticleSystem);
