/*! \file   vanaheimr-config.cpp
	\author Gregory Diamos
	\date   Sunday January 24, 2010
	\brief  The source file for the vanaheimr-config tool.
*/

// Hydrazine Includes
#include <hydrazine/interface/ArgumentParser.h>

// Generated Includes
#include <configure.h>

// Standard Library Includes
#include <string>

namespace vanaheimr
{

class VanaheimrConfig
{
private:
	std::string _flags() const;
	std::string _version() const;
	std::string _prefix() const;
	std::string _libs() const;
	std::string _cxxflags() const;
	std::string _includedir() const;
	std::string _libdir() const;
	std::string _bindir() const;
	std::string _tracelibs() const;

public:
	bool version;
	bool flags;
	bool prefix;
	bool libs;
	bool includedir;
	bool libdir;
	bool bindir;
	bool trace;

public:
	VanaheimrConfig();
	std::string string() const; 
};

std::string VanaheimrConfig::_flags() const
{
	#ifdef VANAHEIMR_CXXFLAGS
	return VANAHEIMR_CXXFLAGS;
	#else
	assertM(false, "Unknown CXX flags, is vanaheimr configured?.");
	#endif
}

std::string VanaheimrConfig::_version() const
{
	#ifdef VERSION
	return VERSION;
	#else
	assertM(false, "Unknown version, is vanaheimr configured?.");
	#endif
}

std::string VanaheimrConfig::_prefix() const
{
	#ifdef VANAHEIMR_PREFIX_PATH
	return VANAHEIMR_PREFIX_PATH;
	#else
	assertM(false, "Unknown prefix, is vanaheimr configured?.");
	#endif
}

std::string VanaheimrConfig::_libs() const
{
	#ifdef VANAHEIMR_LDFLAGS
	return VANAHEIMR_LDFLAGS;
	#else
	assertM(false, "Unknown lib flags, is vanaheimr configured?.");
	#endif
}

std::string VanaheimrConfig::_includedir() const
{
	#ifdef VANAHEIMR_INCLUDE_PATH
	return VANAHEIMR_INCLUDE_PATH;
	#else
	assertM(false, "Unknown include dir, is vanaheimr configured?.");
	#endif
}

std::string VanaheimrConfig::_libdir() const
{
	#ifdef VANAHEIMR_LIB_PATH
	return VANAHEIMR_LIB_PATH;
	#else
	assertM(false, "Unknown lib dir, is vanaheimr configured?.");
	#endif
}

std::string VanaheimrConfig::_bindir() const
{
	#ifdef VANAHEIMR_BIN_PATH
	return VANAHEIMR_BIN_PATH;
	#else
	assertM(false, "Unknown bin dir, is vanaheimr configured?.");
	#endif
}

std::string VanaheimrConfig::string() const
{
	std::string result;
	if( version )
	{
		result += _version() + " ";
	}
	if( flags )
	{
		result += _flags() + " ";
	}
	if( prefix )
	{
		result += _prefix() + " ";
	}
	if( libs )
	{
		result += _libs() + " ";
	}
	if( includedir )
	{
		result += _includedir() + " ";
	}
	if( libdir )
	{
		result += _libdir() + " ";
	}
	if( bindir )
	{
		result += _bindir() + " ";
	}
	
	return result + "\n";
}

VanaheimrConfig::VanaheimrConfig()
{

}
	
}

int main(int argc, char** argv)
{
	hydrazine::ArgumentParser parser(argc, argv);
	vanaheimr::VanaheimrConfig config;
	
	parser.parse( "-l", "--libs", config.libs, false,
		"Libraries needed to link against Vanaheimr." );
	parser.parse( "-x", "--cxxflags", config.flags, false,
		"C++ flags for programs that include Vanaheimr headers." );
	parser.parse( "-L", "--libdir", config.libdir,  false,
		"Directory containing Vanaheimr libraries." );
	parser.parse( "-i", "--includedir", config.includedir, false,
		"Directory containing Vanaheimr headers." );
	parser.parse( "-b", "--bindir", config.bindir, false,
		"Directory containing Vanaheimr executables." );
	parser.parse( "-v", "--version", config.version, false,
		"Print Vanaheimr version." );
	parser.parse( "-p", "--prefix", config.prefix, false,
		"Print the install prefix." );
	parser.parse();

	std::cout << config.string();
	
	return 0;
}


