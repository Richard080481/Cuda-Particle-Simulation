/**
 * CUDA Particle System: Life-cycle & Spatial Partitioning
 * Features:
 * 1. Grid-accelerated collision (O(N) complexity)
 * 2. Mouse explosion physics (Inverse Square Law)
 * 3. Lifetime system (Color evolution from Hot Yellow to Cold Blue)
 */

#define CCCL_IGNORE_DEPRECATED_CPP_DIALECT 
#define NOMINMAX 

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <iostream>
#include <vector>

 // --- Constants ---
const int numParticles = 8192; // Total number of particles
const int gridRes      = 64;         // Grid resolution (64x64 cells)
const float dt         = 0.003f;        // Physics time step

struct Particle
{
    float2 pos;
    float2 vel;
    float life;      // Lifetime: 1.0 (Newborn) -> 0.0 (Dead)
    float fadeRate;  // How fast the particle ages
};

// Global interaction variables
float2 mousePos   = { 0.0f, 0.0f };
bool mouseClicked = false;

// --- GPU Kernels ---

// Simple Pseudo-Random Generator
__device__ float get_rand(int index, float seed)
{
    size_t s = index + (__float_as_int(seed));
    s = (s ^ 61) ^ (s >> 16);
    s *= 9;
    s = s ^ (s >> 4);
    s *= 0x27d4eb2d;
    s = s ^ (s >> 15);
    return (float)(s & 0xFFFFFF) / 16777216.0f;
}

// Map 2D position to 1D grid index
__device__ int getGridIdx(float2 pos, int res)
{
    int x = (int)((pos.x + 1.0f) * 0.5f * res);
    int y = (int)((pos.y + 1.0f) * 0.5f * res);
    x = max(0, min(res - 1, x));
    y = max(0, min(res - 1, y));
    return y * res + x;
}

// Assign grid keys to each particle
__global__ void calcGridKeys(Particle* particles, int* gridKeys, int* indices, int n, int res)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        gridKeys[i] = getGridIdx(particles[i].pos, res);
        indices[i] = i;
    }
}

// Find the start/end index of each cell in the sorted particle array
__global__ void buildOffsets(int* gridKeys, int* gridOffsets, int n, int numCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        int key = gridKeys[i];
        if (i == 0)
        {
            gridOffsets[key] = 0;
        }
        else if (key != gridKeys[i - 1])
        {
            gridOffsets[key] = i;
        }
        if (i == n - 1)
        {
            gridOffsets[numCells] = n;
        }
    }
}

__global__ void updatePhysics(Particle* oldParts,
                              Particle* newParts,
                              int* gridOffsets,
                              int* sortedIndices,
                              float2* vbo_pos,
                              float3* vbo_col,
                              int n,
                              int res,
                              float dt,
                              float time,
                              float2 mPos,
                              bool mClick)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }

    int originalIdx = sortedIndices[i];
    Particle p      = oldParts[originalIdx];
    float2 force    = { 0.0f, 0.0f };

    // 1. Life Cycle & Reset Mechanism
    p.life -= p.fadeRate * dt;
    if (p.life <= 0.0f)
    {
        // Reset to center
        p.pos       = { 0.0f, 0.0f };
        float angle = get_rand(originalIdx, time) * 6.283f;
        float speed = 0.2f + get_rand(originalIdx, time * 1.5f) * 0.8f;
        p.vel       = { cosf(angle) * speed, sinf(angle) * speed };
        p.life      = 1.0f;
        p.fadeRate  = 0.1f + get_rand(originalIdx, time * 0.5f) * 0.4f;
    }

    // 2. Spatial Grid Collision (Repulsion)
    int cx = (int)((p.pos.x + 1.0f) * 0.5f * res);
    int cy = (int)((p.pos.y + 1.0f) * 0.5f * res);

    for (int oy = -1; oy <= 1; oy++)
    {
        for (int ox = -1; ox <= 1; ox++)
        {
            int nx = cx + ox;
            int ny = cy + oy;
            if (nx >= 0 && nx < res && ny >= 0 && ny < res)
            {
                int cellIdx = ny * res + nx;
                int start   = gridOffsets[cellIdx];
                int end     = gridOffsets[cellIdx + 1];
                if (start == -1)
                {
                    continue;
                }
                for (int j = start; j < end; j++)
                {
                    int otherIdx = sortedIndices[j];
                    if (originalIdx == otherIdx)
                    {
                        continue;
                    }

                    Particle other = oldParts[otherIdx];
                    float dx       = p.pos.x - other.pos.x;
                    float dy       = p.pos.y - other.pos.y;
                    float dSq      = dx * dx + dy * dy;

                    if (dSq < 0.00015f && dSq > 0.0f)
                    {
                        float d = sqrtf(dSq);
                        force.x += (dx / d) * 0.5f;
                        force.y += (dy / d) * 0.5f;
                    }
                }
            }
        }
    }

    // 3. Mouse Explosion Force
    if (mClick)
    {
        float mDx     = p.pos.x - mPos.x;
        float mDy     = p.pos.y - mPos.y;
        float mDistSq = mDx * mDx + mDy * mDy;

        if (mDistSq < 0.3f)
        {
            float mDist = sqrtf(mDistSq + 0.001f);
            float push  = (1.0f / mDist) * 15.0f;
            force.x    += (mDx / mDist) * push;
            force.y    += (mDy / mDist) * push;
        }
    }

    // 4. Kinematics Integration
    p.vel.x += force.x * dt;
    p.vel.y += force.y * dt;
    p.vel.x *= 0.98f; p.vel.y *= 0.98f; // Air friction/Damping
    p.pos.x += p.vel.x * dt;
    p.pos.y += p.vel.y * dt;

    // Boundary bounce
    if (p.pos.x < -1.0f || p.pos.x > 1.0f)
    {
        p.vel.x *= -0.5f;
    }

    if (p.pos.y < -1.0f || p.pos.y > 1.0f)
    {
        p.vel.y *= -0.5f;
    }

    // 5. Color Evolution: Yellow/Orange (1.0) -> Blue/Purple (0.0)
    vbo_pos[originalIdx] = p.pos;
    vbo_col[originalIdx] =
    {
        p.life,                             // R: fades with life
        p.life * p.life,                    // G: fades faster
        0.4f + (1.0f - p.life) * 0.6f       // B: increases as particle cools down
    };

    newParts[originalIdx] = p;
}

// --- GLFW Callback ---
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        mouseClicked = (action == GLFW_PRESS);
    }
}

int main()
{
    if (!glfwInit())
    {
        return -1;
    }
    GLFWwindow* window = glfwCreateWindow(1024, 1024, "CUDA Particle Life Cycle", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    // Initial Host Setup
    std::vector<Particle> h_parts(numParticles);
    for (int i = 0; i < numParticles; i++)
    {
        h_parts[i].pos      = { 0.0f, 0.0f };
        h_parts[i].life     = (float)i / numParticles;
        h_parts[i].fadeRate = 0.2f + (rand() % 100 / 100.0f) * 0.3f;
    }

    // GPU Memory Allocation
    Particle* d_old, * d_new;
    int* d_keys, * d_idx, * d_offs;
    cudaMalloc(&d_old, numParticles * sizeof(Particle));
    cudaMalloc(&d_new, numParticles * sizeof(Particle));
    cudaMalloc(&d_keys, numParticles * sizeof(int));
    cudaMalloc(&d_idx, numParticles * sizeof(int));
    cudaMalloc(&d_offs, (gridRes * gridRes + 1) * sizeof(int));
    cudaMemcpy(d_old, h_parts.data(), numParticles * sizeof(Particle), cudaMemcpyHostToDevice);

    // OpenGL VBO Setup
    GLuint vbo, cvbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(float2), NULL, GL_DYNAMIC_DRAW);
    glGenBuffers(1, &cvbo);
    glBindBuffer(GL_ARRAY_BUFFER, cvbo);
    glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(float3), NULL, GL_DYNAMIC_DRAW);

    // CUDA-GL Interoperability
    cudaGraphicsResource* resV, * resC;
    cudaGraphicsGLRegisterBuffer(&resV, vbo, cudaGraphicsRegisterFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&resC, cvbo, cudaGraphicsRegisterFlagsWriteDiscard);

    while (!glfwWindowShouldClose(window))
    {
        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        mousePos.x = (float)(mx / 512.0 - 1.0);
        mousePos.y = (float)(1.0 - my / 512.0);

        int threads = 256;
        int blocks  = (numParticles + threads - 1) / threads;

        // 1. Grid Sorting (Using Thrust)
        calcGridKeys <<<blocks, threads>>> (d_old, d_keys, d_idx, numParticles, gridRes);
        thrust::device_ptr<int> t_keys = thrust::device_pointer_cast(d_keys);
        thrust::device_ptr<int> t_idx = thrust::device_pointer_cast(d_idx);
        thrust::sort_by_key(t_keys, t_keys + numParticles, t_idx);

        // 2. Build Grid Offsets
        cudaMemset(d_offs, -1, (gridRes * gridRes + 1) * sizeof(int));
        buildOffsets <<<blocks, threads>>> (d_keys, d_offs, numParticles, gridRes * gridRes);

        // 3. Map Resources and Run Physics
        float2* dv;
        float3* dc;
        size_t nb;
        cudaGraphicsMapResources(1, &resV, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&dv, &nb, resV);
        cudaGraphicsMapResources(1, &resC, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&dc, &nb, resC);

        updatePhysics <<<blocks, threads>>> (d_old, d_new, d_offs, d_idx, dv, dc, numParticles, gridRes, dt, (float)glfwGetTime(), mousePos, mouseClicked);
        cudaMemcpy(d_old, d_new, numParticles * sizeof(Particle), cudaMemcpyDeviceToDevice);

        cudaGraphicsUnmapResources(1, &resV, 0);
        cudaGraphicsUnmapResources(1, &resC, 0);

        // 4. Rendering
        glClearColor(0.0f, 0.0f, 0.02f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE); // Additive blending for glow effect

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glVertexPointer(2, GL_FLOAT, 0, 0);
        
        glBindBuffer(GL_ARRAY_BUFFER, cvbo);
        glColorPointer(3, GL_FLOAT, 0, 0);

        glPointSize(1.8f);
        glDrawArrays(GL_POINTS, 0, numParticles);

        glDisable(GL_BLEND);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    cudaGraphicsUnregisterResource(resV);
    cudaGraphicsUnregisterResource(resC);
    cudaFree(d_old);
    cudaFree(d_new);
    cudaFree(d_keys);
    cudaFree(d_idx);
    cudaFree(d_offs);
    glfwTerminate();

    return 0;
}