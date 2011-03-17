// Main.cpp : Defines the entry point for the console application.
//

#include "StdAfx.hpp"
#include "Application/SDLGLApplication.hpp"

#ifdef _WIN32
int wmain (int argc, char* argv[])
#else
int main(int argc, char* argv[])
#endif
{

    SDLGLApplication app;
    app.init(argc, argv);
    app.initVideo();

    while(!app.dead())
    {
        app.displayFrame();
        app.fetchEvents();
        SDL_Delay(100);
    }

	return 0;
}

