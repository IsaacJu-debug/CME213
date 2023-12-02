#include <iostream>

// TODO: make a pure function int get_id()
class A
{
};

// TODO: override get_id(), return 1
class B : public A
{
};

// TODO: override get_id(), return 2
class C : public B
{
};


int main()
{   
    C c;

    // print out c's ID
    std::cout << c.get_id() << std::endl;

    // TODO: cast c to type B. Call get_id(). What happens?
    B b;

    // TODO: cast c to type A. Is this possible?

    // TODO: Set some B *bptr to the address of c. Call get_id(). What happens?
    // Do the same for a variable A *aptr. What do you notice?
    B *bptr = nullptr;
    A *aptr = nullptr;
    std::cout << bptr->get_id() << std::endl;
    std::cout << aptr->get_id() << std::endl;
    return 0;
}