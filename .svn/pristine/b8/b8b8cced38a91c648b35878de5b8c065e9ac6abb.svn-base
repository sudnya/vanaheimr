/*!	\file   ImageFile.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Sunday, April 8, 2012
	\brief  The header file for the ImageFile class.
*/

// Standard Library Includes
#include <string>
#include <vector>

#pragma once

// Standard Library Includes
#include <vector>
#include <string>

namespace baldr
{

/*! \brief A simple interface for writing an image to a file */
class ImageFile
{
public:
	class Pixel
	{
	public:
		unsigned int red   : 8;
		unsigned int green : 8;
		unsigned int blue  : 8;
		unsigned int alpha : 8;
	};


public:
	ImageFile(unsigned int width, unsigned int height);
	
public:
	void clear();
	void resize(unsigned int width, unsigned int height);

public:
	void setPixel(unsigned int x, unsigned int y,
		unsigned int red, unsigned int green,
		unsigned int blue, unsigned int alpha);

public:
	void write(const std::string& filename);

private:
	typedef std::vector<Pixel> PixelVector;

private:
	unsigned int _getIndex(unsigned int x, unsigned int y);

private:
	PixelVector  _pixels;
	unsigned int _width;
	unsigned int _height;

};

}


