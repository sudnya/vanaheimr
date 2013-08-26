

#include <archaeopteryx/util/interface/File.h>

#include <archaeopteryx/util/interface/debug.h>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

extern "C" __global__ void deviceMain(const char* filename, void* result,
	const void* data, unsigned int size)
{
	device_report("Testing file access for filename "
		"(%s) with %d bytes\n", filename, size);
	
	archaeopteryx::util::File file(filename);

	file.write(data, size);
	file.read(result, size);
}

