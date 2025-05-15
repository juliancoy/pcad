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
    try {
        // Check if WebGPU is available
        if (!navigator.gpu) {
            document.body.innerHTML = "<h1 style='color:white;text-align:center;padding:2rem'>WebGPU not supported in this browser</h1>";
            return;
        }
        
        // Try to get adapter - this might fail if hardware doesn't support WebGPU
        const adapter = await navigator.gpu.requestAdapter();
        
        // Check if adapter was obtained
        if (!adapter) {
            document.body.innerHTML = "<h1 style='color:white;text-align:center;padding:2rem'>WebGPU Unavailable: Could not get GPU adapter</h1>";
            return;
        }
        
        // Try to get device
        const device = await adapter.requestDevice().catch(error => {
            console.error("Error requesting device:", error);
            return null;
        });
        
        // Check if device was obtained
        if (!device) {
            document.body.innerHTML = "<h1 style='color:white;text-align:center;padding:2rem'>WebGPU Unavailable: Could not get GPU device</h1>";
            return;
        }
        
        // Canvas setup
        const canvas = document.getElementById("webgpu-canvas");
        if (!canvas) {
            document.body.innerHTML = "<h1 style='color:white;text-align:center;padding:2rem'>Error: Canvas element not found</h1>";
            return;
        }
        
        const context = canvas.getContext("webgpu");
        if (!context) {
            document.body.innerHTML = "<h1 style='color:white;text-align:center;padding:2rem'>WebGPU Unavailable: Could not get WebGPU context</h1>";
            return;
        }
        
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
    } catch (error) {
        console.error("WebGPU initialization error:", error);
        document.body.innerHTML = `<h1 style='color:white;text-align:center;padding:2rem'>WebGPU Unavailable: ${error.message}</h1>`;
    }
}

async function createParticleBuffers() {
    try {
        const { device, particleCount, canvas, format } = particleSystem;
        
        if (!device) {
            throw new Error("GPU device not initialized");
        }
        
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
        
        if (countSlider) {
            // Update UI value displays
            countSlider.addEventListener('input', async () => {
                const newCount = parseInt(countSlider.value);
                const countValueElement = document.getElementById('count-value');
                if (countValueElement) {
                    countValueElement.textContent = newCount;
                }
                
                if (particleSystem.running) {
                    cancelAnimationFrame(particleSystem.frameId);
                    particleSystem.running = false;
                }
                
                particleSystem.particleCount = newCount;
                await createParticleBuffers();
                startAnimation();
            });
        }
        
        const uniformData = new Float32Array([
            // Camera position (xyz) and aspect ratio
            0, 0, -10, canvas.width / canvas.height,
            // Time, dt, numCylinders, _pad1
            0, 0.016, 3, 0,
            // Force parameters and padding
            parseFloat(repulsionSlider?.value || "0.5"),
            parseFloat(attractionSlider?.value || "0.5"),
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
        let shaderCode;
        try {
            const response = await fetch('./particles.wgsl');
            if (!response.ok) {
                throw new Error(`Failed to load shader: ${response.status} ${response.statusText}`);
            }
            shaderCode = await response.text();
        } catch (error) {
            console.error("Error loading shader:", error);
            document.body.innerHTML = `<h1 style='color:white;text-align:center;padding:2rem'>Error loading shader: ${error.message}</h1>
                                      <p style='color:white;text-align:center;'>Make sure the file 'particles.wgsl' exists and is accessible.</p>`;
            return;
        }
        
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
        
        // Create render pipeline with quad-based rendering
        particleSystem.pipelines.render = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: shaderModule,
                entryPoint: 'vertexMain',
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragmentMain',
                targets: [{ 
                    format: particleSystem.format,
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add'
                        },
                        alpha: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add'
                        }
                    }
                }]
            },
            primitive: {
                topology: 'triangle-list', // Changed from point-list to triangle-list for quads
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
        
        if (repulsionSlider) {
            // Update UI value displays
            repulsionSlider.addEventListener('input', () => {
                const repulsionValueElement = document.getElementById('repulsion-value');
                if (repulsionValueElement) {
                    repulsionValueElement.textContent = repulsionSlider.value;
                }
                device.queue.writeBuffer(
                    particleSystem.uniformBuffer,
                    32, // Offset to repulsionStrength
                    new Float32Array([parseFloat(repulsionSlider.value)])
                );
            });
        }
        
        if (attractionSlider) {
            attractionSlider.addEventListener('input', () => {
                const attractionValueElement = document.getElementById('attraction-value');
                if (attractionValueElement) {
                    attractionValueElement.textContent = attractionSlider.value;
                }
                device.queue.writeBuffer(
                    particleSystem.uniformBuffer,
                    36, // Offset to attractionStrength
                    new Float32Array([parseFloat(attractionSlider.value)])
                );
            });
        }
    } catch (error) {
        console.error("Error creating particle buffers:", error);
        document.body.innerHTML = `<h1 style='color:white;text-align:center;padding:2rem'>WebGPU Error: ${error.message}</h1>`;
    }
}

function startAnimation() {
    try {
        if (!particleSystem.device || !particleSystem.pipelines.render) {
            console.error("Cannot start animation: WebGPU not properly initialized");
            return;
        }
        
        if (particleSystem.running) return;
        particleSystem.running = true;
        
        let startTime = performance.now() / 1000;
        let lastFrameTime = startTime;
        
        function frame() {
            try {
                const { canvas, device, context, uniformBuffer, depthTexture } = particleSystem;
                        
                // Update time parameters
                const now = performance.now() / 1000;
                const dt = Math.min(0.033, now - lastFrameTime); // Cap at 30fps equivalent
                lastFrameTime = now;
                const time = now - startTime;
                
                // Animate camera position in a circular path
                // Camera animation parameters
                const cameraRadius = 5;
                const cameraHeight = 3;
                const cameraSpeed = 0.02;
            
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
                
                // Render pass with direct rendering to canvas
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
                
                // Draw 6 vertices per particle (2 triangles to form a quad)
                renderPass.draw(particleSystem.particleCount * 6);
                renderPass.end();
                
                // Submit and schedule next frame
                device.queue.submit([commandEncoder.finish()]);
                particleSystem.frameId = requestAnimationFrame(frame);
            } catch (error) {
                console.error("Error in animation frame:", error);
                cancelAnimationFrame(particleSystem.frameId);
                particleSystem.running = false;
                document.body.innerHTML += `<div style='color:red;position:fixed;bottom:20px;left:20px;background:rgba(0,0,0,0.7);padding:10px;'>
                    Animation error: ${error.message}
                </div>`;
            }
        }
        
        particleSystem.frameId = requestAnimationFrame(frame);
    } catch (error) {
        console.error("Error starting animation:", error);
        document.body.innerHTML = `<h1 style='color:white;text-align:center;padding:2rem'>Animation Error: ${error.message}</h1>`;
    }
}

// Initialize everything when the page loads
window.addEventListener('load', async () => {
    try {
        await initParticleSystem();
        if (particleSystem.device) {
            startAnimation();
        }
    } catch (error) {
        console.error("Fatal error initializing WebGPU:", error);
        document.body.innerHTML = `<h1 style='color:white;text-align:center;padding:2rem'>WebGPU Unavailable: ${error.message}</h1>`;
    }
});