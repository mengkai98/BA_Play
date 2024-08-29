#include "BASolver.h"
int main(int argc, char *argv[])
{
    if (argc == 3)
    {

        BASolver(std::string(argv[1]), std::string(argv[2]));
    }

    return 0;
}
