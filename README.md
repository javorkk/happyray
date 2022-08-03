# happyray

This is a CUDA ray tracer that uses uniform or two-level grids as acceleration structures and can render .obj key-frame animations.

The code builds with Visual Studio 2017 and CUDA 8.0 on Windows, and (after some tweaks to the Makefile) with gcc and CUDA 8.0 on Linux.

See scene.cfg about configuring a static scene or a key-frame animation.

Press F1 after launching the application to output the controls in the console.

# X forwarding using Docker on WSL2

# in WSL:
touch ~/.Xauthority
xauth add ${HOST}:0 . $(xxd -l 16 -p /dev/urandom)
xauth list
# result should look like:
IGOR-II/unix:0  MIT-MAGIC-COOKIE-1  9e6238034158c72ba4b08487c0ff0cd0
# in the running docker image:
xauth add ${DISPLAY} MIT-MAGIC-COOKIE-1  9e6238034158c72ba4b08487c0ff0cd0