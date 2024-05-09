#include <util/Log.h>

namespace Log
{
	void Info(const std::string& msg)
	{
		std::cout << "Info: \t" << msg << std::endl;
	}

	void Error(const std::string& msg, const bool exitProgram)
	{
		std::cout << "Error: \t" << msg << std::endl;
		if (exitProgram) { throw std::runtime_error(msg); }
	}

	void Assert(const bool condition, const std::string& msg)
	{
		if (!condition) { Log::Error(msg, true); }
	}
}
