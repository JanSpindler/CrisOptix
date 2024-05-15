#include <util/read_file.h>
#include <fstream>
#include <util/Log.h>

std::vector<char> ReadFile(const std::string& fileName)
{
    std::ifstream input(fileName, std::ios::binary);
    Log::Assert(input.is_open());

    input.seekg(0, std::ios::end);
    const size_t size = static_cast<size_t>(input.tellg());
    input.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    input.read(buffer.data(), buffer.size());

    return buffer;
}
