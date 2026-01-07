/**
 * CUDA Particle System: 3D Life-cycle & Spatial Partitioning
 * Features:
 * 1. 3D Grid-accelerated collision (O(N) complexity)
 * 2. Mouse explosion physics in 3D space
 * 3. Lifetime system (Color evolution from Hot Yellow to Cold Blue)
 * 4. Interactive 3D camera with mouse controls
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
#include <cmath>

 // --- Constants ---
const int numParticles = 8192; // Increased for 3D
const int gridRes = 32;    // 32x32x32 grid
const float dt = 0.003f;

struct Particle
{
    float3 pos;
    float3 vel;
    float life;
    float fadeRate;
};

// Camera state
struct Camera
{
    float3 pos;
    float3 target;
    float3 up;
    float pitch;
    float yaw;
    float distance;
} camera = { {0.0f, 0.0f, 3.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.0f, 0.0f, 3.0f };

// Mouse state
float2 mousePos = { 0.0f, 0.0f };
float2 lastMouse = { 0.0f, 0.0f };
bool mouseClicked = false;
bool mouseDragging = false;

// --- GPU Kernels ---

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

// Map 3D position to 1D grid index
__device__ int getGridIdx(float3 pos, int res)
{
    int x = (int)((pos.x + 1.0f) * 0.5f * res);
    int y = (int)((pos.y + 1.0f) * 0.5f * res);
    int z = (int)((pos.z + 1.0f) * 0.5f * res);
    x = max(0, min(res - 1, x));
    y = max(0, min(res - 1, y));
    z = max(0, min(res - 1, z));
    return z * res * res + y * res + x;
}

__global__ void calcGridKeys(Particle* particles, int* gridKeys, int* indices, int n, int res)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        gridKeys[i] = getGridIdx(particles[i].pos, res);
        indices[i] = i;
    }
}

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
    float3* vbo_pos,
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
    Particle p = oldParts[originalIdx];
    float3 force = { 0.0f, 0.0f, 0.0f };

    // 1. Life Cycle & Reset
    p.life -= p.fadeRate * dt;
    if (p.life <= 0.0f)
    {
        p.pos = { 0.0f, 0.0f, 0.0f };
        float theta = get_rand(originalIdx, time) * 6.283f;
        float phi = acosf(2.0f * get_rand(originalIdx, time * 1.3f) - 1.0f);
        float speed = 0.3f + get_rand(originalIdx, time * 1.5f) * 0.7f;
        p.vel = {
            sinf(phi) * cosf(theta) * speed,
            sinf(phi) * sinf(theta) * speed,
            cosf(phi) * speed
        };
        p.life = 1.0f;
        p.fadeRate = 0.1f + get_rand(originalIdx, time * 0.5f) * 0.4f;
    }

    // 2. 3D Spatial Grid Collision
    int cx = (int)((p.pos.x + 1.0f) * 0.5f * res);
    int cy = (int)((p.pos.y + 1.0f) * 0.5f * res);
    int cz = (int)((p.pos.z + 1.0f) * 0.5f * res);

    for (int oz = -1; oz <= 1; oz++)
    {
        for (int oy = -1; oy <= 1; oy++)
        {
            for (int ox = -1; ox <= 1; ox++)
            {
                int nx = cx + ox;
                int ny = cy + oy;
                int nz = cz + oz;
                if (nx >= 0 && nx < res && ny >= 0 && ny < res && nz >= 0 && nz < res)
                {
                    int cellIdx = nz * res * res + ny * res + nx;
                    int start = gridOffsets[cellIdx];
                    int end = gridOffsets[cellIdx + 1];
                    if (start == -1) continue;

                    for (int j = start; j < end; j++)
                    {
                        int otherIdx = sortedIndices[j];
                        if (originalIdx == otherIdx) continue;

                        Particle other = oldParts[otherIdx];
                        float dx = p.pos.x - other.pos.x;
                        float dy = p.pos.y - other.pos.y;
                        float dz = p.pos.z - other.pos.z;
                        float dSq = dx * dx + dy * dy + dz * dz;

                        if (dSq < 0.0002f && dSq > 0.0f)
                        {
                            float d = sqrtf(dSq);
                            float repulsion = 0.4f;
                            force.x += (dx / d) * repulsion;
                            force.y += (dy / d) * repulsion;
                            force.z += (dz / d) * repulsion;
                        }
                    }
                }
            }
        }
    }

    // 3. Mouse Explosion (in 3D, projects to XY plane)
    if (mClick)
    {
        float mDx = p.pos.x - mPos.x;
        float mDy = p.pos.y - mPos.y;
        float mDistSq = mDx * mDx + mDy * mDy;

        if (mDistSq < 0.4f)
        {
            float mDist = sqrtf(mDistSq + 0.001f);
            float push = (1.0f / mDist) * 12.0f;
            force.x += (mDx / mDist) * push;
            force.y += (mDy / mDist) * push;
            // Add outward Z force
            force.z += (p.pos.z > 0 ? 1.0f : -1.0f) * push * 0.5f;
        }
    }

    // 4. Central attraction (weak)
    float centerDist = sqrtf(p.pos.x * p.pos.x + p.pos.y * p.pos.y + p.pos.z * p.pos.z);
    if (centerDist > 0.1f)
    {
        force.x -= p.pos.x * 0.05f;
        force.y -= p.pos.y * 0.05f;
        force.z -= p.pos.z * 0.05f;
    }

    // 5. Integration
    p.vel.x += force.x * dt;
    p.vel.y += force.y * dt;
    p.vel.z += force.z * dt;
    p.vel.x *= 0.985f;
    p.vel.y *= 0.985f;
    p.vel.z *= 0.985f;
    p.pos.x += p.vel.x * dt;
    p.pos.y += p.vel.y * dt;
    p.pos.z += p.vel.z * dt;

    // Boundary bounce
    float bounce = 0.6f;
    if (p.pos.x < -1.0f || p.pos.x > 1.0f) p.vel.x *= -bounce;
    if (p.pos.y < -1.0f || p.pos.y > 1.0f) p.vel.y *= -bounce;
    if (p.pos.z < -1.0f || p.pos.z > 1.0f) p.vel.z *= -bounce;

    // Clamp to bounds
    p.pos.x = fmaxf(-1.0f, fminf(1.0f, p.pos.x));
    p.pos.y = fmaxf(-1.0f, fminf(1.0f, p.pos.y));
    p.pos.z = fmaxf(-1.0f, fminf(1.0f, p.pos.z));

    // 6. Color Evolution with depth
    float depthFactor = (p.pos.z + 1.0f) * 0.5f; // 0 to 1
    vbo_pos[originalIdx] = p.pos;
    vbo_col[originalIdx] = {
        p.life * (0.8f + depthFactor * 0.2f),
        p.life * p.life * (0.7f + depthFactor * 0.3f),
        0.3f + (1.0f - p.life) * 0.5f + depthFactor * 0.2f
    };

    newParts[originalIdx] = p;
}

// --- Camera Functions ---
void updateCamera()
{
    camera.pos.x = camera.distance * cos(camera.pitch) * cos(camera.yaw);
    camera.pos.y = camera.distance * sin(camera.pitch);
    camera.pos.z = camera.distance * cos(camera.pitch) * sin(camera.yaw);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        mouseClicked = (action == GLFW_PRESS);
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        if (action == GLFW_PRESS)
        {
            mouseDragging = true;
            glfwGetCursorPos(window, (double*)&lastMouse.x, (double*)&lastMouse.y);
        }
        else
        {
            mouseDragging = false;
        }
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.distance -= (float)yoffset * 0.3f;
    camera.distance = fmaxf(1.5f, fminf(8.0f, camera.distance));
    updateCamera();
}

int main()
{
    if (!glfwInit()) return -1;

    GLFWwindow* window = glfwCreateWindow(1200, 1200, "3D CUDA Particle Life Cycle", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // Initial particles
    std::vector<Particle> h_parts(numParticles);
    for (int i = 0; i < numParticles; i++)
    {
        h_parts[i].pos = { 0.0f, 0.0f, 0.0f };
        h_parts[i].vel = { 0.0f, 0.0f, 0.0f };
        h_parts[i].life = (float)i / numParticles;
        h_parts[i].fadeRate = 0.2f + (rand() % 100 / 100.0f) * 0.3f;
    }

    // GPU Memory
    Particle* d_old, * d_new;
    int* d_keys, * d_idx, * d_offs;
    cudaMalloc(&d_old, numParticles * sizeof(Particle));
    cudaMalloc(&d_new, numParticles * sizeof(Particle));
    cudaMalloc(&d_keys, numParticles * sizeof(int));
    cudaMalloc(&d_idx, numParticles * sizeof(int));
    cudaMalloc(&d_offs, (gridRes * gridRes * gridRes + 1) * sizeof(int));
    cudaMemcpy(d_old, h_parts.data(), numParticles * sizeof(Particle), cudaMemcpyHostToDevice);

    // OpenGL VBOs
    GLuint vbo, cvbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(float3), NULL, GL_DYNAMIC_DRAW);
    glGenBuffers(1, &cvbo);
    glBindBuffer(GL_ARRAY_BUFFER, cvbo);
    glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(float3), NULL, GL_DYNAMIC_DRAW);

    // CUDA-GL Interop
    cudaGraphicsResource* resV, * resC;
    cudaGraphicsGLRegisterBuffer(&resV, vbo, cudaGraphicsRegisterFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&resC, cvbo, cudaGraphicsRegisterFlagsWriteDiscard);

    updateCamera();

    while (!glfwWindowShouldClose(window))
    {
        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        mousePos.x = (float)(mx / 600.0 - 1.0);
        mousePos.y = (float)(1.0 - my / 600.0);

        // Camera rotation with right mouse button
        if (mouseDragging)
        {
            float dx = (float)(mx - lastMouse.x);
            float dy = (float)(my - lastMouse.y);
            camera.yaw += dx * 0.005f;
            camera.pitch += dy * 0.005f;
            camera.pitch = fmaxf(-1.5f, fminf(1.5f, camera.pitch));
            updateCamera();
            lastMouse.x = (float)mx;
            lastMouse.y = (float)my;
        }

        int threads = 256;
        int blocks = (numParticles + threads - 1) / threads;

        // Grid sorting
        calcGridKeys << <blocks, threads >> > (d_old, d_keys, d_idx, numParticles, gridRes);
        thrust::device_ptr<int> t_keys = thrust::device_pointer_cast(d_keys);
        thrust::device_ptr<int> t_idx = thrust::device_pointer_cast(d_idx);
        thrust::sort_by_key(t_keys, t_keys + numParticles, t_idx);

        cudaMemset(d_offs, -1, (gridRes * gridRes * gridRes + 1) * sizeof(int));
        buildOffsets << <blocks, threads >> > (d_keys, d_offs, numParticles, gridRes * gridRes * gridRes);

        // Map and update
        float3* dv, * dc;
        size_t nb;
        cudaGraphicsMapResources(1, &resV, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&dv, &nb, resV);
        cudaGraphicsMapResources(1, &resC, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&dc, &nb, resC);

        updatePhysics << <blocks, threads >> > (d_old, d_new, d_offs, d_idx, dv, dc,
            numParticles, gridRes, dt,
            (float)glfwGetTime(), mousePos, mouseClicked);
        cudaMemcpy(d_old, d_new, numParticles * sizeof(Particle), cudaMemcpyDeviceToDevice);

        cudaGraphicsUnmapResources(1, &resV, 0);
        cudaGraphicsUnmapResources(1, &resC, 0);

        // Render with gradient background
        glClearColor(0.05f, 0.08f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Draw background gradient
        glDisable(GL_DEPTH_TEST);
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        glBegin(GL_QUADS);
        glColor3f(0.02f, 0.04f, 0.10f); // Dark blue at top
        glVertex2f(-1.0f, 1.0f);
        glVertex2f(1.0f, 1.0f);
        glColor3f(0.08f, 0.12f, 0.20f); // Lighter blue at bottom
        glVertex2f(1.0f, -1.0f);
        glVertex2f(-1.0f, -1.0f);
        glEnd();

        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glEnable(GL_DEPTH_TEST);

        // Setup 3D projection
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        float aspect = 1.0f;
        float fov = 60.0f * 3.14159f / 180.0f;
        float f = 1.0f / tanf(fov / 2.0f);
        float zm[16] = {
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (10.0f + 0.1f) / (0.1f - 10.0f), -1,
            0, 0, (2 * 10.0f * 0.1f) / (0.1f - 10.0f), 0
        };
        glMultMatrixf(zm);

        // Setup camera view
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        float cx = camera.pos.x, cy = camera.pos.y, cz = camera.pos.z;
        float tx = camera.target.x, ty = camera.target.y, tz = camera.target.z;
        float ux = camera.up.x, uy = camera.up.y, uz = camera.up.z;

        float fwd[3] = { tx - cx, ty - cy, tz - cz };
        float flen = sqrtf(fwd[0] * fwd[0] + fwd[1] * fwd[1] + fwd[2] * fwd[2]);
        fwd[0] /= flen; fwd[1] /= flen; fwd[2] /= flen;

        float side[3] = {
            fwd[1] * uz - fwd[2] * uy,
            fwd[2] * ux - fwd[0] * uz,
            fwd[0] * uy - fwd[1] * ux
        };
        float slen = sqrtf(side[0] * side[0] + side[1] * side[1] + side[2] * side[2]);
        side[0] /= slen; side[1] /= slen; side[2] /= slen;

        float up[3] = {
            side[1] * fwd[2] - side[2] * fwd[1],
            side[2] * fwd[0] - side[0] * fwd[2],
            side[0] * fwd[1] - side[1] * fwd[0]
        };

        float vm[16] = {
            side[0], up[0], -fwd[0], 0,
            side[1], up[1], -fwd[1], 0,
            side[2], up[2], -fwd[2], 0,
            -(side[0] * cx + side[1] * cy + side[2] * cz),
            -(up[0] * cx + up[1] * cy + up[2] * cz),
            (fwd[0] * cx + fwd[1] * cy + fwd[2] * cz), 1
        };
        glMultMatrixf(vm);

        // Draw reference grid to show 3D space
        glDisable(GL_DEPTH_TEST);
        glLineWidth(1.0f);
        glBegin(GL_LINES);
        glColor4f(0.15f, 0.2f, 0.3f, 0.5f);
        for (int i = -5; i <= 5; i++)
        {
            float pos = i * 0.4f;
            // Grid on XZ plane
            glVertex3f(pos, -1.0f, -2.0f);
            glVertex3f(pos, -1.0f, 2.0f);
            glVertex3f(-2.0f, -1.0f, pos);
            glVertex3f(2.0f, -1.0f, pos);
        }
        glEnd();

        // Draw coordinate axes
        glLineWidth(2.0f);
        glBegin(GL_LINES);
        // X axis - Red
        glColor3f(0.6f, 0.2f, 0.2f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(1.5f, 0.0f, 0.0f);
        // Y axis - Green
        glColor3f(0.2f, 0.6f, 0.2f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 1.5f, 0.0f);
        // Z axis - Blue
        glColor3f(0.2f, 0.2f, 0.6f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 1.5f);
        glEnd();

        // Draw boundary box [-1, 1] in all dimensions
        glLineWidth(2.0f);
        glColor4f(0.3f, 0.4f, 0.5f, 0.8f);
        glBegin(GL_LINES);
        // Bottom face (Y = -1)
        glVertex3f(-1.0f, -1.0f, -1.0f); glVertex3f(1.0f, -1.0f, -1.0f);
        glVertex3f(1.0f, -1.0f, -1.0f); glVertex3f(1.0f, -1.0f, 1.0f);
        glVertex3f(1.0f, -1.0f, 1.0f); glVertex3f(-1.0f, -1.0f, 1.0f);
        glVertex3f(-1.0f, -1.0f, 1.0f); glVertex3f(-1.0f, -1.0f, -1.0f);
        // Top face (Y = 1)
        glVertex3f(-1.0f, 1.0f, -1.0f); glVertex3f(1.0f, 1.0f, -1.0f);
        glVertex3f(1.0f, 1.0f, -1.0f); glVertex3f(1.0f, 1.0f, 1.0f);
        glVertex3f(1.0f, 1.0f, 1.0f); glVertex3f(-1.0f, 1.0f, 1.0f);
        glVertex3f(-1.0f, 1.0f, 1.0f); glVertex3f(-1.0f, 1.0f, -1.0f);
        // Vertical edges
        glVertex3f(-1.0f, -1.0f, -1.0f); glVertex3f(-1.0f, 1.0f, -1.0f);
        glVertex3f(1.0f, -1.0f, -1.0f); glVertex3f(1.0f, 1.0f, -1.0f);
        glVertex3f(1.0f, -1.0f, 1.0f); glVertex3f(1.0f, 1.0f, 1.0f);
        glVertex3f(-1.0f, -1.0f, 1.0f); glVertex3f(-1.0f, 1.0f, 1.0f);
        glEnd();

        glEnable(GL_DEPTH_TEST);

        // Draw particles
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glVertexPointer(3, GL_FLOAT, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, cvbo);
        glColorPointer(3, GL_FLOAT, 0, 0);

        glPointSize(2.0f);
        glDrawArrays(GL_POINTS, 0, numParticles);

        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
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
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &cvbo);
    glfwTerminate();

    return 0;
}