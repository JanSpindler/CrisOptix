#pragma once

#include <string>
#include <iostream>

namespace Log
{
	static void Info(const std::string& msg)
	{
		std::cout << "Info: \t" << msg << std::endl;
	}

	static void Error(const std::string& msg, const bool exitProgram = true)
	{
		std::cout << "Error: \t" << msg << std::endl;
		if (exitProgram) { throw std::runtime_error(msg); }
	}
}
