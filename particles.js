const particleSystem = {
    device: null,
    canvas: null,
    context: null,
    pipelines: {
        compute: null,
        render: null
    },
    buffers: null,
    bindGroups: {
        compute: null,
        render: null
    },
    uniformBuffer: null,
    depthTexture: null,
    particleCount: 1024,
    running: false,
    frameId: null
};

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
    
    function resizeCanvas() {
        const devicePixelRatio = window.devicePixelRatio || 1;
        const canvas = particleSystem.canvas;
    
        canvas.width = window.innerWidth * devicePixelRatio;
        canvas.height = window.innerHeight * devicePixelRatio;
    
        // ✅ Only update the uniform buffer if it already exists
        if (particleSystem.uniformBuffer) {
            particleSystem.device.queue.writeBuffer(
                particleSystem.uniformBuffer,
                12, // Offset to aspectRatio
                new Float32Array([canvas.width / canvas.height])
            );
        }
    
        // ✅ Recreate depth texture to match canvas size
        if (particleSystem.depthTexture) {
            particleSystem.depthTexture.destroy();
        }
    
        particleSystem.depthTexture = particleSystem.device.createTexture({
            size: [canvas.width, canvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });
    }
    
    
    window.addEventListener('resize', resizeCanvas);
    
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ 
        device, 
        format, 
        alphaMode: "premultiplied" 
    });
    
    // Initialize WebGPU if not already done
    if (!particleSystem.device) {
        particleSystem.device = device;
        particleSystem.canvas = canvas;
        particleSystem.context = context;
        particleSystem.format = format;
    }
    
    resizeCanvas();
    // Create or update particle buffers
    await createParticleBuffers();
}

async function createParticleBuffers() {
    const { device, particleCount, canvas, format } = particleSystem;
    
    // Initialize particles
    const positions = new Float32Array(particleCount * 3);
    const velocities = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    
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
    
    // Create or recreate buffers
    if (particleSystem.buffers) {
        particleSystem.buffers.position.destroy();
        particleSystem.buffers.velocity.destroy();
        particleSystem.buffers.color.destroy();
        if (particleSystem.buffers.cylinder) {
            particleSystem.buffers.cylinder.destroy();
        }
    }
    
    particleSystem.buffers = {
        position: device.createBuffer({
            size: positions.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        }),
        velocity: device.createBuffer({
            size: velocities.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        }),
        color: device.createBuffer({
            size: colors.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        })
    };

    // Initialize buffer data
    new Float32Array(particleSystem.buffers.position.getMappedRange()).set(positions);
    particleSystem.buffers.position.unmap();
    
    new Float32Array(particleSystem.buffers.velocity.getMappedRange()).set(velocities);
    particleSystem.buffers.velocity.unmap();
    
    new Float32Array(particleSystem.buffers.color.getMappedRange()).set(colors);
    particleSystem.buffers.color.unmap();
    
    // Cylinder data: centerX, centerY, centerZ, radius, height (flattened array)
    const cylinderData = new Float32Array([
        0, 0, 0, 1.0, 2.0,    // Cylinder 1: center(0,0,0), radius 1.0, height 2.0
        0, 0, 0, 1.0, 2.0,    // Cylinder 1: center(0,0,0), radius 1.0, height 2.0
        0, 0, 0, 1.0, 2.0,    // Cylinder 1: center(0,0,0), radius 1.0, height 2.0
    ]);
    
    particleSystem.buffers.cylinder = device.createBuffer({
        size: cylinderData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Float32Array(particleSystem.buffers.cylinder.getMappedRange()).set(cylinderData);
    particleSystem.buffers.cylinder.unmap();
    
    // Create uniform buffer for camera and simulation parameters
    // Get UI control elements
    const countSlider = document.getElementById('particle-count');
    const repulsionSlider = document.getElementById('repulsion-strength');
    const attractionSlider = document.getElementById('attraction-strength');
    
    // Update UI value displays
    countSlider.addEventListener('input', async () => {
        const newCount = parseInt(countSlider.value);
        document.getElementById('count-value').textContent = newCount;
        
        if (particleSystem.running) {
            cancelAnimationFrame(particleSystem.frameId);
            particleSystem.running = false;
        }
        
        particleSystem.particleCount = newCount;
        await createParticleBuffers();
        startAnimation();
    });
    
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
    
    // Create or recreate uniform buffer
    if (particleSystem.uniformBuffer) {
        particleSystem.uniformBuffer.destroy();
    }
    
    particleSystem.uniformBuffer = device.createBuffer({
        size: uniformData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Float32Array(particleSystem.uniformBuffer.getMappedRange()).set(uniformData);
    particleSystem.uniformBuffer.unmap();
    
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
    particleSystem.pipelines.compute = device.createComputePipeline({
        layout: 'auto',
        compute: { 
            module: shaderModule, 
            entryPoint: 'computeMain' 
        }
    });
    
    // Create render pipeline
    particleSystem.pipelines.render = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
            module: shaderModule,
            entryPoint: 'vertexMain',
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fragmentMain',
            targets: [{ format: particleSystem.format }]
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
    
    // Create or recreate depth texture
    if (particleSystem.depthTexture) {
        particleSystem.depthTexture.destroy();
    }
    
    particleSystem.depthTexture = device.createTexture({
        size: [canvas.width, canvas.height],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
    
    // Create bind group for compute pipeline
    particleSystem.bindGroups.compute = device.createBindGroup({
        layout: particleSystem.pipelines.compute.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: particleSystem.buffers.position }},
            { binding: 1, resource: { buffer: particleSystem.buffers.velocity }},
            { binding: 2, resource: { buffer: particleSystem.buffers.color }},
            { binding: 3, resource: { buffer: particleSystem.buffers.cylinder }},
            { binding: 4, resource: { buffer: particleSystem.uniformBuffer }}
        ]
    });
    
    // Create bind group for render pipeline
    particleSystem.bindGroups.render = device.createBindGroup({
        layout: particleSystem.pipelines.render.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: particleSystem.buffers.position }},
            { binding: 2, resource: { buffer: particleSystem.buffers.color }},
            { binding: 4, resource: { buffer: particleSystem.uniformBuffer }}
        ]
    });
    
    // Update UI value displays
    repulsionSlider.addEventListener('input', () => {
        document.getElementById('repulsion-value').textContent = repulsionSlider.value;
        device.queue.writeBuffer(
            particleSystem.uniformBuffer,
            32, // Offset to repulsionStrength
            new Float32Array([parseFloat(repulsionSlider.value)])
        );
    });
    
    attractionSlider.addEventListener('input', () => {
        document.getElementById('attraction-value').textContent = attractionSlider.value;
        device.queue.writeBuffer(
            particleSystem.uniformBuffer,
            36, // Offset to attractionStrength
            new Float32Array([parseFloat(attractionSlider.value)])
        );
    });
}

function startAnimation() {
    if (particleSystem.running) return;
    particleSystem.running = true;
    
    let startTime = performance.now() / 1000;
    let lastFrameTime = startTime;
    
    function frame() {
        const { canvas, device, context, uniformBuffer, depthTexture } = particleSystem;
                
        // Update time parameters
        const now = performance.now() / 1000;
        const dt = Math.min(0.033, now - lastFrameTime); // Cap at 30fps equivalent
        lastFrameTime = now;
        const time = now - startTime;
        
        // Animate camera position in a circular path
        // Camera animation parameters
        const cameraRadius = 10;
        const cameraHeight = 3;
        const cameraSpeed = 0.3;
    
        const cameraX = cameraRadius * Math.sin(time * cameraSpeed);
        const cameraZ = cameraRadius * Math.cos(time * cameraSpeed);
        const cameraY = cameraHeight * Math.sin(time * cameraSpeed * 0.5);
        
        // Update uniform buffer with new camera position and time
        device.queue.writeBuffer(
            particleSystem.uniformBuffer,
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
        computePass.setPipeline(particleSystem.pipelines.compute);
        computePass.setBindGroup(0, particleSystem.bindGroups.compute);
        computePass.dispatchWorkgroups(Math.ceil(particleSystem.particleCount / 64));
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
                view: particleSystem.depthTexture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store'
            }
        });
        
        renderPass.setPipeline(particleSystem.pipelines.render);
        renderPass.setBindGroup(0, particleSystem.bindGroups.render);
        renderPass.draw(particleSystem.particleCount);
        renderPass.end();
        
        // Submit and schedule next frame
        device.queue.submit([commandEncoder.finish()]);
        particleSystem.frameId = requestAnimationFrame(frame);
    }
    
    particleSystem.frameId = requestAnimationFrame(frame);
}

// Initialize everything when the page loads
window.addEventListener('load', async () => {
    await initParticleSystem();
    startAnimation();
});