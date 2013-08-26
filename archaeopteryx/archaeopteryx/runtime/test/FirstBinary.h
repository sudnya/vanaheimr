/*! \file FirstBinary.h 
    \date   Saturday October 23, 2011
    \author Sudnya Padalikar
            <mailsudnya@gmail.com>
    \brief The header file for a test program to write a binary file for saxpy.
*/

#pragma once

class Header
{
    public:
        unsigned int dataPages;
        unsigned int codePages;
        unsigned int symbols;
        unsigned int strings;
};

class SymbolTableEntry
{
    public:
        /*! \brief The type of symbol */
        unsigned int type;
        /*! \brief The offset in the string table of the name */
        unsigned int stringTableOffset;
        /*! \brief The page id it is stored in */
        unsigned int pageId;
        /*! \brief The offset within the page */
        unsigned int pageOffset;
        /*! \brief The set of attributes */
        long long unsigned int attributes;
};

