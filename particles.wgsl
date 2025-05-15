struct Uniforms {
    cameraPos: vec3<f32>,
    aspectRatio: f32,
    time: f32,
    dt: f32, 
    numCylinders: f32,
    _pad1: f32,
    repulsionStrength: f32,
    attractionStrength: f32,
    _pad2: f32,
};

struct Cylinder {
    center: vec3<f32>,
    radius: f32,
    height: f32,
};

// === COMPUTE SHADER RESOURCES ===
@group(0) @binding(0) var<storage, read_write> positions: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> colors: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read> cylinders: array<f32>;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

// === VERTEX SHADER RESOURCES ===
@group(0) @binding(0) var<storage, read> vertexPositions: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> vertexColors: array<vec3<f32>>;
@group(0) @binding(4) var<uniform> vertexUniforms: Uniforms;

const MAX_CYLINDERS = 10;
var<private> cylinderCache: array<Cylinder, MAX_CYLINDERS>;

fn sdf_finite_cylinder(p: vec3<f32>, cylinder: Cylinder) -> f32 {
    // Vector from cylinder center to point
    let localP = p - cylinder.center;
    
    // Project point onto cylinder axis (y-axis)
    let dxz = length(vec2(localP.x, localP.z)) - cylinder.radius;
    let dy = abs(localP.y) - cylinder.height * 0.5;
    
    let outside = max(dxz, dy);
    let inside = min(max(dxz, dy), 0.0);
    
    return outside + inside;
}

fn sdf_cylinder_gradient(p: vec3<f32>, cylinder: Cylinder) -> vec3<f32> {
    let eps = 0.01;
    let dx = sdf_finite_cylinder(p + vec3(eps, 0.0, 0.0), cylinder) -
            sdf_finite_cylinder(p - vec3(eps, 0.0, 0.0), cylinder);
    let dy = sdf_finite_cylinder(p + vec3(0.0, eps, 0.0), cylinder) -
            sdf_finite_cylinder(p - vec3(0.0, eps, 0.0), cylinder);
    let dz = sdf_finite_cylinder(p + vec3(0.0, 0.0, eps), cylinder) -
            sdf_finite_cylinder(p - vec3(0.0, 0.0, eps), cylinder);
    return normalize(vec3(dx, dy, dz));
}

@compute @workgroup_size(64)
fn computeMain(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&positions)) {
        return;
    }

    // Cache cylinders for efficiency
    let numCylinders = i32(uniforms.numCylinders);
    for (var c = 0; c < numCylinders; c++) {
        let idx = c * 5; // Each cylinder takes 5 floats: centerX, centerY, centerZ, radius, height
        cylinderCache[c] = Cylinder(
            vec3(cylinders[idx], cylinders[idx+1], cylinders[idx+2]),
            cylinders[idx+3],
            cylinders[idx+4]
        );
    }

    var p = positions[i];
    var v = velocities[i];

    // Apply attraction to cylinder surfaces
    var totalForce = vec3(0.0);
    var nearestDist = 1000.0;
    var nearestColor = vec3(1.0, 1.0, 1.0);
    
    // Process each cylinder
    for (var c = 0; c < numCylinders; c++) {
        let cylinder = cylinderCache[c];
        let d = sdf_finite_cylinder(p, cylinder);
        let g = sdf_cylinder_gradient(p, cylinder);
        
        let targetDist = 0.05; // desired distance from surface
        let distError = d - targetDist;
        
        // Add attraction force toward cylinder surface
        totalForce += -g * distError * uniforms.attractionStrength;
        
        // Track nearest cylinder for coloring
        if (abs(d) < abs(nearestDist)) {
            nearestDist = d;
            
            // Generate color based on cylinder and distance
            let hue = f32(c) / f32(numCylinders);
            let saturation = clamp(1.0 - abs(d) * 0.2, 0.5, 1.0);
            
            // Simple HSV to RGB conversion
            let h = hue * 6.0;
            let i = floor(h);
            let f = h - i;
            let p = 1.0 - saturation;
            let q = 1.0 - saturation * f;
            let t = 1.0 - saturation * (1.0 - f);
            
            if (i < 1.0) {
                nearestColor = vec3(1.0, t, p);
            } else if (i < 2.0) {
                nearestColor = vec3(q, 1.0, p);
            } else if (i < 3.0) {
                nearestColor = vec3(p, 1.0, t);
            } else if (i < 4.0) {
                nearestColor = vec3(p, q, 1.0);
            } else if (i < 5.0) {
                nearestColor = vec3(t, p, 1.0);
            } else {
                nearestColor = vec3(1.0, p, q);
            }
        }
    }
    
    // Add repulsion from other particles
    let repulsionRange = 0.5;
    let softening = 0.1;
    
    // For optimization, we'll only sample a subset of particles
    let sampleRate = 4u;
    let particleCount = arrayLength(&positions);
    
    for (var j = 0u; j < particleCount; j += sampleRate) {
        if (i == j) { continue; }
        
        let pj = positions[j];
        let dir = p - pj;
        let dist = length(dir);
        
        if (dist < 0.0001 || dist > repulsionRange) { continue; }
        
        // Inverse square repulsion with a softening factor
        totalForce += normalize(dir) * uniforms.repulsionStrength / (dist * dist + softening);
    }
    
    // Add a slight attraction to origin to prevent particles from drifting too far
    let originAttractionStrength = 0.01;
    let distToOrigin = length(p);
    if (distToOrigin > 5.0) {
        totalForce -= p * originAttractionStrength;
    }
    
    // Add some noise for natural movement
    let noise = vec3(
        sin(uniforms.time * 5.0 + f32(i) * 0.1) * 0.01,
        cos(uniforms.time * 4.0 + f32(i) * 0.2) * 0.01,
        sin(uniforms.time * 3.0 + f32(i) * 0.3) * 0.01
    );
    totalForce += noise;
    
    // Apply damping
    v *= 0.98;
    
    // Update velocity and position
    v += totalForce * uniforms.dt;
    p += v * uniforms.dt;
    
    // Update buffers
    velocities[i] = v;
    positions[i] = p;
    
    // Update color based on nearest cylinder
    colors[i] = mix(colors[i], nearestColor, 0.05);
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var output: VertexOutput;
    
    let pos = vertexPositions[vertexIndex];
    
    // Simple perspective projection
    let fov = 45.0 * 3.14159 / 180.0;
    let near = 0.1;
    let far = 100.0;
    
    // Camera-relative position
    let camPos = vertexUniforms.cameraPos;
    let cameraDistance = length(pos - camPos);
    
    // View matrix (simple lookAt)
    let cameraTarget = vec3(0.0, 0.0, 0.0);
    let up = vec3(0.0, 1.0, 0.0);
    
    let cameraDir = normalize(cameraTarget - camPos);
    let cameraRight = normalize(cross(cameraDir, up));
    let cameraUp = cross(cameraRight, cameraDir);
    
    // Apply view transform
    let viewSpacePos = vec3(
        dot(pos - camPos, cameraRight),
        dot(pos - camPos, cameraUp),
        dot(pos - camPos, cameraDir)
    );
    
    // Apply perspective projection
    let f = 1.0 / tan(fov / 2.0);
    let aspect = vertexUniforms.aspectRatio;
    
    let clipSpacePos = vec4(
        viewSpacePos.x * f / aspect,
        viewSpacePos.y * f,
        (viewSpacePos.z * (far + near) - 2.0 * far * near) / (viewSpacePos.z * (far - near)),
        viewSpacePos.z
    );
    
    output.position = clipSpacePos;
    output.color = vertexColors[vertexIndex];
    
    return output;
}

@fragment
fn fragmentMain(
    @location(0) color: vec3<f32>,
    @builtin(position) fragPos: vec4<f32>
) -> @location(0) vec4<f32> {
    // Simple point rendering
    return vec4<f32>(color, 1.0);
}
