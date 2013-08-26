EnsureSConsVersion(1,2)

import os

import inspect
import platform
import re
import subprocess
from SCons import SConf

def getTools():
    result = []
    if os.name == 'nt':
        result = ['default', 'msvc']
    elif os.name == 'posix':
        result = ['default', 'c++', 'g++']
    else:
        result = ['default']

    return result;


OldEnvironment = Environment;


# this dictionary maps the name of a compiler program to a dictionary mapping the name of
# a compiler switch of interest to the specific switch implementing the feature
gCompilerOptions = {
        'gcc' : {'warn_all' : '-Wall',
            'warn_errors' : '-Werror',
            'optimization' : '-O2', 'debug' : '-g', 
            'exception_handling' : '', 'standard': ''},
        'g++' : {'warn_all' : '-Wall',
            'warn_errors' : '-Werror',
            'optimization' : '-O2', 'debug' : '-g', 
            'exception_handling' : '', 'standard': ''},
        'c++' : {'warn_all' : '-Wall',
            'warn_errors' : '-Werror',
            'optimization' : '-O2', 'debug' : '-g',
            'exception_handling' : '',
            'standard': ['-stdlib=libc++', '-std=c++0x', '-pthread']},
        'cl'  : {'warn_all' : '/Wall',
                 'warn_errors' : '/WX', 
                 'optimization' : ['/Ox', '/MD', '/Zi', '/DNDEBUG'], 
                 'debug' : ['/Zi', '/Od', '/D_DEBUG', '/RTC1', '/MDd'], 
                 'exception_handling': '/EHsc', 
                 'standard': ['/GS', '/GR', '/Gd', '/fp:precise',
                     '/Zc:wchar_t','/Zc:forScope', '/DYY_NO_UNISTD_H']}
    }


# this dictionary maps the name of a linker program to a dictionary mapping the name of
# a linker switch of interest to the specific switch implementing the feature
gLinkerOptions = {
        'gcc'  : {'debug' : ''},
        'g++'  : {'debug' : ''},
        'c++'  : {'debug' : ''},
        'link' : {'debug' : '/debug'}
    }

def getCFLAGS(mode, warn, warnings_as_errors, CC):
    result = []
    if mode == 'release':
        # turn on optimization
        result.append(gCompilerOptions[CC]['optimization'])
    elif mode == 'debug':
        # turn on debug mode
        result.append(gCompilerOptions[CC]['debug'])
        result.append('-DBALDR_DEBUG')

    if warn:
        # turn on all warnings
        result.append(gCompilerOptions[CC]['warn_all'])

    if warnings_as_errors:
        # treat warnings as errors
        result.append(gCompilerOptions[CC]['warn_errors'])

    result.append(gCompilerOptions[CC]['standard'])

    return result

def getCXXFLAGS(mode, warn, warnings_as_errors, CXX):
    result = []
    if mode == 'release':
        # turn on optimization
        result.append(gCompilerOptions[CXX]['optimization'])
    elif mode == 'debug':
        # turn on debug mode
        result.append(gCompilerOptions[CXX]['debug'])
    # enable exception handling
    result.append(gCompilerOptions[CXX]['exception_handling'])

    if warn:
        # turn on all warnings
        result.append(gCompilerOptions[CXX]['warn_all'])

    if warnings_as_errors:
        # treat warnings as errors
        result.append(gCompilerOptions[CXX]['warn_errors'])

    result.append(gCompilerOptions[CXX]['standard'])

    return result

def getLINKFLAGS(mode, LINK):
    result = []
    if mode == 'debug':
        # turn on debug mode
        result.append(gLinkerOptions[LINK]['debug'])

    return result

def getExtraLibs():
    if os.name == 'nt':
        return []
    else:
        return ['-lpng']

def importEnvironment():
    env = {  }
    
    if 'PATH' in os.environ:
        env['PATH'] = os.environ['PATH']
    
    if 'CXX' in os.environ:
        env['CXX'] = os.environ['CXX']
    
    if 'CC' in os.environ:
        env['CC'] = os.environ['CC']
    
    if 'TMP' in os.environ:
        env['TMP'] = os.environ['TMP']
    
    if 'LIBRARY_PATH' in os.environ:
        env['LIBRARY_PATH'] = os.environ['LIBRARY_PATH']
    
    if 'LD_LIBRARY_PATH' in os.environ:
        env['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']
    
    if 'CPLUS_INCLUDE_PATH' in os.environ:
        env['CPLUS_INCLUDE_PATH'] = os.environ['CPLUS_INCLUDE_PATH']

    return env

def Environment():
    vars = Variables()

    # add a variable to handle RELEASE/DEBUG mode
    vars.Add(EnumVariable('mode', 'Release versus debug mode', 'debug',
        allowed_values = ('release', 'debug')))

    # add a variable to handle warnings
    vars.Add(BoolVariable('Wall', 'Enable all compilation warnings', 1))
    
    # shared or static libraries
    libraryDefault = 'shared'
    
    vars.Add(EnumVariable('library', 'Build shared or static library',
        libraryDefault, allowed_values = ('shared', 'static')))
    
    # add a variable to treat warnings as errors
    vars.Add(BoolVariable('Werror', 'Treat warnings as errors', 1))

    # add a variable to determine the install path
    default_install_path = '/usr/local'
    
    if 'BALDR_INSTALL_PATH' in os.environ:
        default_install_path = os.environ['BALDR_INSTALL_PATH']
        
    vars.Add(PathVariable('install_path', 'The vanaheimr install path',
        default_install_path))

    # create an Environment
    env = OldEnvironment(ENV = importEnvironment(), \
        tools = getTools(), variables = vars)

    # always link with the c++ compiler
    if os.name != 'nt':
        env['LINK'] = env['CXX']
    
    # get the absolute path to the directory containing
    # this source file
    thisFile = inspect.getabsfile(Environment)
    thisDir = os.path.dirname(thisFile)

    # get C compiler switches
    env.AppendUnique(CFLAGS = getCFLAGS(env['mode'], env['Wall'], \
        env['Werror'], env.subst('$CC')))
    env.AppendUnique(CXXFLAGS = getCXXFLAGS(env['mode'], env['Wall'], \
        env['Werror'], env.subst('$CXX')))

    # get linker switches
    env.AppendUnique(LINKFLAGS = getLINKFLAGS(env['mode'], env.subst('$LINK')))

    # Install paths
    if 'install' in COMMAND_LINE_TARGETS:
        env.Replace(INSTALL_PATH = os.path.abspath(env['install_path']))
    else:
        env.Replace(INSTALL_PATH = os.path.abspath('.'))

    # get libc++
    if env['CXX'] == 'c++':
        env.AppendUnique(CPPPATH = getLibCXXPaths()[0])
    
    # set extra libs 
    env.Replace(EXTRA_LIBS=getExtraLibs())
    env.Prepend(CPPPATH = os.path.dirname(thisDir))
    env.AppendUnique(LIBPATH = os.path.abspath('.'))
    
    # we need librt on linux
    if platform.system() == 'Linux':
        env.AppendUnique(EXTRA_LIBS = ['-lrt']) 

    print env['EXTRA_LIBS']

    # generate help text
    Help(vars.GenerateHelpText(env))

    return env

