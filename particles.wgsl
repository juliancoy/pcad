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
            vec3(cylinders[idx], cylinders[idx + 1], cylinders[idx + 2]),
            cylinders[idx + 3],
            cylinders[idx + 4]
        );
    }

    var p = positions[i];
    var v = velocities[i];

    var totalForce = vec3(0.0);
    var nearestDist = 1000.0;
    var nearestColor = vec3(1.0, 1.0, 1.0);

    // Compute total gradient for this particle
    var gi = vec3(0.0);
    for (var c = 0; c < numCylinders; c++) {
        let cylinder = cylinderCache[c];
        let d = sdf_finite_cylinder(p, cylinder);
        let g = sdf_cylinder_gradient(p, cylinder);
        gi += g;

        // Add attraction force toward cylinder surface
        totalForce += -g * d * uniforms.attractionStrength;

        // Track nearest cylinder for coloring
        if (abs(d) < abs(nearestDist)) {
            nearestDist = d;
            nearestColor = vec3(1.0, 0.0, 1.0);
        }
    }
    gi = normalize(gi);

    // Add repulsion from other particles
    let repulsionRange = 0.8;
    let softening = 0.1;
    let sampleRate = 1u;
    let particleCount = arrayLength(&positions);

    for (var j = 0u; j < particleCount; j += sampleRate) {
        if (i == j) { continue; }

        let pj = positions[j];
        let dir = p - pj;
        let dist = length(dir);

        if (dist < 0.0001 || dist > repulsionRange) { continue; }

        // Compute gradient normal for particle j
        var gj = vec3(0.0);
        for (var c = 0; c < numCylinders; c++) {
            gj += sdf_cylinder_gradient(pj, cylinderCache[c]);
        }
        gj = normalize(gj);

        // Stronger repulsion when normals are similar
        let normalDot = dot(gi, gj);
        let normalWeight = pow(normalDot, 2.0)*5;

        // Inverse square repulsion with normal weighting
        let repulsion = normalize(dir) * uniforms.repulsionStrength * normalWeight / (dist * dist + softening);
        totalForce += repulsion;
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

// Output to vertex shader - quad per particle
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) quadPos: vec2<f32>,
};

// Vertex shader using orthogonal projection
@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var output: VertexOutput;
    
    // Extract particle index and vertex offset within the quad
    let particleIndex = vertexIndex / 6u;  // 6 vertices per quad (2 triangles)
    let vertexOffset = vertexIndex % 6u;
    
    if (particleIndex >= arrayLength(&vertexPositions)) {
        // Return a degenerate vertex if out of bounds
        output.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        output.color = vec3<f32>(0.0);
        output.quadPos = vec2<f32>(0.0);
        return output;
    }
    
    // Get particle position and color
    let particlePos = vertexPositions[particleIndex];
    let particleColor = vertexColors[particleIndex];
    
    // Camera parameters
    let camPos = vertexUniforms.cameraPos;
    let aspect = vertexUniforms.aspectRatio;
    
    // Camera setup (simple lookAt)
    let cameraTarget = vec3(0.0, 0.0, 0.0);
    let up = vec3(0.0, 1.0, 0.0);
    let cameraDir = normalize(cameraTarget - camPos);
    let cameraRight = normalize(cross(cameraDir, up));
    let cameraUp = cross(cameraRight, cameraDir);
    
    // Calculate view-space position
    let viewSpacePos = vec3(
        dot(particlePos - camPos, cameraRight),
        dot(particlePos - camPos, cameraUp),
        dot(particlePos - camPos, cameraDir)
    );
    
    // Orthographic projection parameters
    let orthoSize = 3.0; // Controls the zoom level (smaller = more zoomed in)
    let near = -100.0;
    let far = 100.0;
    
    // Quad size - this is our dilation factor (3x3 pixels in screen space)
    let quadSize = 0.02; // Adjust based on orthoSize to get proper 3x3 pixel dilation
    
    // Offset for each vertex of the quad
    // We're creating 2 triangles (6 vertices) to form a quad
    var quadOffset = vec2<f32>(0.0, 0.0);
    var localQuadPos = vec2<f32>(0.0, 0.0);
    
    switch(vertexOffset) {
        case 0u: { // Triangle 1, Vertex 1
            quadOffset = vec2<f32>(-1.0, -1.0);
            localQuadPos = vec2<f32>(-1.0, -1.0);
        }
        case 1u: { // Triangle 1, Vertex 2
            quadOffset = vec2<f32>(1.0, -1.0);
            localQuadPos = vec2<f32>(1.0, -1.0);
        }
        case 2u: { // Triangle 1, Vertex 3
            quadOffset = vec2<f32>(-1.0, 1.0);
            localQuadPos = vec2<f32>(-1.0, 1.0);
        }
        case 3u: { // Triangle 2, Vertex 1
            quadOffset = vec2<f32>(1.0, -1.0);
            localQuadPos = vec2<f32>(1.0, -1.0);
        }
        case 4u: { // Triangle 2, Vertex 2
            quadOffset = vec2<f32>(1.0, 1.0);
            localQuadPos = vec2<f32>(1.0, 1.0);
        }
        case 5u: { // Triangle 2, Vertex 3
            quadOffset = vec2<f32>(-1.0, 1.0);
            localQuadPos = vec2<f32>(-1.0, 1.0);
        }
        default: {}
    }
    
    // Offset the view space position
    let offsetViewSpacePos = vec3(
        viewSpacePos.x + quadOffset.x * quadSize,
        viewSpacePos.y + quadOffset.y * quadSize,
        viewSpacePos.z
    );
    
    // Apply orthographic projection
    let clipSpacePos = vec4(
        offsetViewSpacePos.x / (orthoSize * aspect),
        offsetViewSpacePos.y / orthoSize,
        (offsetViewSpacePos.z - (far + near) / 2.0) / ((far - near) / 2.0),
        1.0
    );
    
    output.position = clipSpacePos;
    output.color = particleColor;
    output.quadPos = localQuadPos;
    
    return output;
}

@fragment
fn fragmentMain(
    @location(0) color: vec3<f32>,
    @location(1) quadPos: vec2<f32>
) -> @location(0) vec4<f32> {
    // Calculate distance from center of quad
    let distFromCenter = length(quadPos);
    
    // Create a circular point by discarding fragments outside radius
    if (distFromCenter > 1.0) {
        discard;
    }
    
    // Optional: create softer edges with alpha falloff
    let alpha = 1.0 - smoothstep(0.7, 1.0, distFromCenter);
    
    // Return color with alpha for smoother blending
    return vec4<f32>(color, alpha);
}