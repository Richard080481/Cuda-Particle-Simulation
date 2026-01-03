# Cuda-Particle-Simulation# CUDA Particle Simulation: Mouse-Triggered Explosion

A high-performance GPU-accelerated particle simulation using NVIDIA CUDA and OpenGL, featuring 8,192 particles with real-time physics, spatial grid-based collision detection, and interactive mouse-triggered explosions.

The simulation uses CUDA kernels for parallel particle updates, maintains elastic collisions between particles through a grid acceleration structure, applies inverse-square-law forces on mouse click events, and enforces perpetual motion with speed constraints.

 Particles spawn at the screen center with radial velocity distribution, bounce off boundaries, and dynamically change color based on speed.
 
 **Requirements**: NVIDIA GPU with CUDA 11.0+, GLFW 3.x, GLEW, and C++17 compiler.
 
 **Build**: `nvcc kernel.cu -o particle_sim -lglfw3 -lglew32 -lopengl32 -std=c++17`.
 
 **Usage**: Run the executable and left-click anywhere on the 1024×1024 window to trigger particle explosions.
 
 The implementation uses CUDA-OpenGL interop for zero-copy rendering and Thrust library for efficient GPU sorting of particles by spatial grid keys, achieving 60+ FPS on modern GPUs.
