/**
 * CUDA Particle Simulation: Mouse-Triggered Explosion
 * Features: Static initial position, Mouse click explosion, and Grid-accelerated collisions.
 */

#define CCCL_IGNORE_DEPRECATED_CPP_DIALECT
#define NOMINMAX

#include <GL/glew.h>
#ifdef APIENTRY
#undef APIENTRY
#endif
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

 // --- Constants ---
const int numParticles = 32768 / 4;
const int gridRes = 64;
const float dt = 0.003f;

// Global variables for mouse interaction
float2 mousePos = { 0.0f, 0.0f };
bool mouseClicked = false;

struct Particle {
    float2 pos;
    float2 vel;
};

// --- GPU Kernels ---

__device__ float get_random_noise(int index, float seed) {
    size_t s = index + (__float_as_int(seed));
    s = (s ^ 61) ^ (s >> 16);
    s *= 9;
    s = s ^ (s >> 4);
    s *= 0x27d4eb2d;
    s = s ^ (s >> 15);
    return (float)(s & 0xFFFFFF) / 16777216.0f;
}

__device__ int getGridIdx(float2 pos, int res) {
    int x = (int)((pos.x + 1.0f) * 0.5f * res);
    int y = (int)((pos.y + 1.0f) * 0.5f * res);
    x = (int)fmaxf(0.0f, fminf((float)res - 1.0f, (float)x));
    y = (int)fmaxf(0.0f, fminf((float)res - 1.0f, (float)y));
    return y * res + x;
}

__global__ void calcGridKeys(Particle* particles, int* gridKeys, int* indices, int n, int res) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        gridKeys[i] = getGridIdx(particles[i].pos, res);
        indices[i] = i;
    }
}

__global__ void buildOffsets(int* gridKeys, int* gridOffsets, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int key = gridKeys[i];
        if (i == 0) gridOffsets[key] = 0;
        else if (key != gridKeys[i - 1]) gridOffsets[key] = i;
        if (i == n - 1) gridOffsets[gridRes * gridRes] = n;
    }
}

__global__ void updatePhysics(Particle* oldParts, Particle* newParts, int* gridOffsets, int* sortedIndices,
    float2* vbo_ptr, float3* color_ptr, int n, int res, float dt, float time,
    float2 mPos, bool mClick) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int originalIdx = sortedIndices[i];
        Particle p = oldParts[originalIdx];
        float2 force = { 0.0f, 0.0f };

        // 1. COLLISION DETECTION (Bounce back logic)
        int cx = (int)((p.pos.x + 1.0f) * 0.5f * res);
        int cy = (int)((p.pos.y + 1.0f) * 0.5f * res);

        for (int oy = -1; oy <= 1; oy++) {
            for (int ox = -1; ox <= 1; ox++) {
                int nx = cx + ox; int ny = cy + oy;
                if (nx >= 0 && nx < res && ny >= 0 && ny < res) {
                    int cellIdx = ny * res + nx;
                    int start = gridOffsets[cellIdx];
                    int end = gridOffsets[cellIdx + 1];
                    for (int j = start; j < end; j++) {
                        int otherIdx = sortedIndices[j];
                        if (originalIdx == otherIdx) continue;
                        Particle other = oldParts[otherIdx];
                        float dx = p.pos.x - other.pos.x;
                        float dy = p.pos.y - other.pos.y;
                        float dSq = dx * dx + dy * dy;
                        // Elastic Repulsion Force (Bounce)
                        if (dSq < 0.00022f && dSq > 0.0f) {
                            float dist = sqrtf(dSq);
                            float push = (0.015f - dist) * 250.0f;
                            force.x += (dx / dist) * push;
                            force.y += (dy / dist) * push;
                        }
                    }
                }
            }
        }

        // 2. MOUSE EXPLOSION LOGIC
        if (mClick) {
            float mDx = p.pos.x - mPos.x;
            float mDy = p.pos.y - mPos.y;
            float mDistSq = mDx * mDx + mDy * mDy;
            if (mDistSq < 0.2f) { // Explosion Range
                float mDist = sqrtf(mDistSq + 0.001f);
                float mPush = (1.0f / mDist) * 15.0f; // Inverse square law shockwave
                force.x += (mDx / mDist) * mPush;
                force.y += (mDy / mDist) * mPush;
            }
        }

        // 3. MOTION INTEGRATION (Perpetual)
        p.vel.x += force.x * dt;
        p.vel.y += force.y * dt;

        // Speed safeguards
        float speedSq = p.vel.x * p.vel.x + p.vel.y * p.vel.y;
        if (speedSq < 0.36f) { // Min speed 0.6
            float angle = get_random_noise(originalIdx, time) * 6.28f;
            p.vel.x = cosf(angle) * 0.6f; p.vel.y = sinf(angle) * 0.6f;
        }
        if (speedSq > 25.0f) { // Max speed 5.0
            float ratio = 5.0f / sqrtf(speedSq);
            p.vel.x *= ratio; p.vel.y *= ratio;
        }

        p.pos.x += p.vel.x * dt; p.pos.y += p.vel.y * dt;

        // Wall Bounce
        if (p.pos.x < -1.0f || p.pos.x > 1.0f) p.vel.x *= -1.0f;
        if (p.pos.y < -1.0f || p.pos.y > 1.0f) p.vel.y *= -1.0f;
        p.pos.x = fmaxf(-1.0f, fminf(1.0f, p.pos.x));
        p.pos.y = fmaxf(-1.0f, fminf(1.0f, p.pos.y));

        newParts[originalIdx] = p;
        vbo_ptr[originalIdx] = p.pos;
        color_ptr[originalIdx] = { 0.4f + speedSq * 0.02f, 0.6f, 1.0f };
    }
}

// GLFW Mouse Callback
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        mouseClicked = (action == GLFW_PRESS);
    }
}

int main() {
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(1024, 1024, "CUDA Click Explosion", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    // Initial Position: FIXED AT MIDDLE (0,0)
    Particle* h_parts = (Particle*)malloc(numParticles * sizeof(Particle));
    for (int i = 0; i < numParticles; i++) {
        h_parts[i].pos = { 0.0f, 0.0f };
        float angle = (i / (float)numParticles) * 6.28f;
        h_parts[i].vel = { cosf(angle) * 2.0f, sinf(angle) * 2.0f };
    }

    Particle* d_old, * d_new;
    int* d_keys, * d_idx, * d_offs;
    cudaMalloc(&d_old, numParticles * sizeof(Particle));
    cudaMalloc(&d_new, numParticles * sizeof(Particle));
    cudaMalloc(&d_keys, numParticles * sizeof(int));
    cudaMalloc(&d_idx, numParticles * sizeof(int));
    cudaMalloc(&d_offs, (gridRes * gridRes + 1) * sizeof(int));
    cudaMemcpy(d_old, h_parts, numParticles * sizeof(Particle), cudaMemcpyHostToDevice);

    GLuint vbo, cvbo;
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo); glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(float2), NULL, GL_DYNAMIC_DRAW);
    glGenBuffers(1, &cvbo); glBindBuffer(GL_ARRAY_BUFFER, cvbo); glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(float3), NULL, GL_DYNAMIC_DRAW);

    cudaGraphicsResource* resV, * resC;
    cudaGraphicsGLRegisterBuffer(&resV, vbo, cudaGraphicsRegisterFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&resC, cvbo, cudaGraphicsRegisterFlagsWriteDiscard);

    while (!glfwWindowShouldClose(window)) {
        // Get Mouse Cursor Position
        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        mousePos.x = (float)(mx / 512.0 - 1.0);
        mousePos.y = (float)(1.0 - my / 512.0);

        float time = (float)glfwGetTime();
        int threads = 256;
        int blocks = (numParticles + threads - 1) / threads;

        calcGridKeys << <blocks, threads >> > (d_old, d_keys, d_idx, numParticles, gridRes);
        thrust::device_ptr<int> t_keys = thrust::device_pointer_cast(d_keys);
        thrust::device_ptr<int> t_idx = thrust::device_pointer_cast(d_idx);
        thrust::sort_by_key(t_keys, t_keys + numParticles, t_idx);

        cudaMemset(d_offs, 0, (gridRes * gridRes + 1) * sizeof(int));
        buildOffsets << <blocks, threads >> > (d_keys, d_offs, numParticles);

        float2* dv; float3* dc; size_t nb;
        cudaGraphicsMapResources(1, &resV, 0); cudaGraphicsResourceGetMappedPointer((void**)&dv, &nb, resV);
        cudaGraphicsMapResources(1, &resC, 0); cudaGraphicsResourceGetMappedPointer((void**)&dc, &nb, resC);

        updatePhysics << <blocks, threads >> > (d_old, d_new, d_offs, d_idx, dv, dc, numParticles, gridRes, dt, time, mousePos, mouseClicked);
        cudaMemcpy(d_old, d_new, numParticles * sizeof(Particle), cudaMemcpyDeviceToDevice);

        cudaGraphicsUnmapResources(1, &resV, 0); cudaGraphicsUnmapResources(1, &resC, 0);

        glClearColor(0.0f, 0.0f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_COLOR_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, vbo); glVertexPointer(2, GL_FLOAT, 0, 0);
        glBindBuffer(GL_ARRAY_BUFFER, cvbo); glColorPointer(3, GL_FLOAT, 0, 0);
        glPointSize(1.5f); glDrawArrays(GL_POINTS, 0, numParticles);
        glfwSwapBuffers(window); glfwPollEvents();
    }
    return 0;
}