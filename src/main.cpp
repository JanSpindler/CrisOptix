#include <iostream>
#include <Window.h>

int main()
{
    std::cout << "Hello there" << std::endl;

    Window::Init(800, 600, true, "CrisOptix");

    while (!Window::IsClosed())
    {
        Window::Update();
    }

    Window::Destroy();

    return 0;
}
