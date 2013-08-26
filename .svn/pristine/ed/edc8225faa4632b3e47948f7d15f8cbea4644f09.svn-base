/*!	\file   ImageFile.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Sunday, April 8, 2012
	\brief  The source file for the ImageFile class.
*/

// Baldr Includes
#include <baldr/include/ImageFile.h>

// LibPNG Includes
#include <png.h>

// Standard Library Includes
#include <cstdio>
#include <stdexcept>

namespace baldr
{

ImageFile::ImageFile(unsigned int width, unsigned int height)
{
	resize(width, height);
}

void ImageFile::clear()
{
	_width  = 0;
	_height = 0;

	_pixels.clear();
}

void ImageFile::resize(unsigned int width, unsigned int height)
{
	_width  = width;
	_height = height;

	_pixels.resize(width * height);
}

void ImageFile::setPixel(unsigned int x, unsigned int y,
	unsigned int red, unsigned int green,
	unsigned int blue, unsigned int alpha)
{
	Pixel p;

	p.red   = red;
	p.green = green;
	p.blue  = blue;
	p.alpha = alpha;

	_pixels[_getIndex(x, y)] = p;
}

void ImageFile::write(const std::string& filename)
{
	FILE* file = std::fopen(filename.c_str(), "wb");

	if(file == 0)
	{
		throw std::runtime_error("Could not open image file "
			+ filename + " for writing.");
	}
	
	png_structp writeStructure = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);

	if(writeStructure == 0)
	{
		throw std::runtime_error("Creating PNG write structure failed.");
	}
	
	png_info* pngInfo = png_create_info_struct(writeStructure);

	if(pngInfo == 0)
	{
		throw std::runtime_error("Creatig PNG info structure failed.");
	}

	png_init_io(writeStructure, file);

	png_set_IHDR(writeStructure, pngInfo, _width, _height, 8, PNG_COLOR_TYPE_RGBA,
		PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	png_write_info(writeStructure, pngInfo);

	for(unsigned int i = 0; i < _height; ++i)
	{
		png_write_row(writeStructure, (png_bytep)&_pixels[_getIndex(0,i)]);
	}

	png_write_end(writeStructure, 0);

	std::fclose(file);
}

unsigned int ImageFile::_getIndex(unsigned int x, unsigned int y)
{
	return y * _height + x;
}

}


