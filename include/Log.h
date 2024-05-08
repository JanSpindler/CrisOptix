#pragma once

#include <string>
#include <iostream>

namespace Log
{
	void Info(const std::string& msg);
	void Error(const std::string& msg, const bool exitProgram = true);
}
