/*! \file   LexerRule.cpp
	\date   April 28, 2013
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The source file for the LexerRule class.
*/

// Vanaheimr Includes
#include <vanaheimr/parser/interface/LexerRule.h>

// Standard Library Includes
#include <vector>

namespace vanaheimr
{

namespace parser
{

LexerRule::LexerRule(const std::string& regex)
{
	typedef std::vector<size_t> PositionVector; 

	bool escape = false;

	PositionVector wildcardPositions;

	// record wildcards
	for(auto character : regex)
	{
		if(character == '\\')
		{
			escape = true;
		
			continue;
		}
		
		if(escape)
		{
			_regex.push_back(character);
			
			escape = false;
			
			continue;
		}
		
		if(character == '*')
		{
			wildcardPositions.push_back(_regex.size());
		}
		
		_regex.push_back(character);
	}
	
	for(auto position : wildcardPositions)
	{
		_wildcards.insert(_regex.begin() + position);
	}
}

bool LexerRule::canMatchWithBegin(const std::string& text) const
{
	return _matchWithBegin(text.begin(), text.end());
}

bool LexerRule::canMatchWithEnd(const std::string& text) const
{
	for(auto beginPosition = text.begin();
		beginPosition != text.end(); ++beginPosition)
	{
		if(_matchWithEnd(beginPosition, text.end())) return true;
	}
	
	return false;
}

bool LexerRule::canOnlyMatchWithBegin(const std::string& text) const
{
	if(!canMatchWithBegin(text)) return false;
	
	return isExactMatch(text) || !canMatchWithEnd(text);
}

bool LexerRule::canOnlyMatchWithEnd(const std::string& text) const
{
	if(!canMatchWithEnd(text)) return false;
	
	return isExactMatch(text) || !canMatchWithBegin(text);
}
	
bool LexerRule::canMatch(const std::string& text) const
{
	for(auto beginPosition = text.begin();
		beginPosition != text.end(); ++beginPosition)
	{
		std::string::const_iterator position = beginPosition;
	
		if(_match(position, beginPosition, text.end())) return true;
	}

	return false;
}

const std::string& LexerRule::toString() const
{
	return _regex;
}

LexerRule::iterator LexerRule::begin()
{
	return _regex.begin();
}

LexerRule::const_iterator LexerRule::begin() const
{
	return _regex.begin();
}

LexerRule::iterator LexerRule::end()
{
	return _regex.end();
}

LexerRule::const_iterator LexerRule::end() const
{
	return _regex.end();
}

LexerRule::reverse_iterator LexerRule::rbegin()
{
	return _regex.rbegin();
}

LexerRule::const_reverse_iterator LexerRule::rbegin() const
{
	return _regex.rbegin();
}

LexerRule::reverse_iterator LexerRule::rend()
{
	return _regex.rend();
}

LexerRule::const_reverse_iterator LexerRule::rend() const
{
	return _regex.rend();
}

bool LexerRule::empty() const
{
	return _regex.empty();
}

size_t LexerRule::size() const
{
	return _regex.size();
}

bool LexerRule::isExactMatch(const std::string& text) const
{
	auto textMatchEnd = text.begin();
	auto ruleMatchEnd = begin();
	
	if(!_match(textMatchEnd, ruleMatchEnd, text.begin(), text.end(),
		begin(), end()))
	{
		return false;
	}
	
	return textMatchEnd == text.end() && ruleMatchEnd == end();
}

bool LexerRule::_match(const_iterator& matchEnd,
	const_iterator& matchRuleEnd,
	const_iterator begin, const_iterator end,
	const_iterator ruleBegin, const_iterator ruleEnd) const
{
	auto ruleCharacter = ruleBegin;
	
	for( ; ruleCharacter != ruleEnd; )
	{
		auto ruleNextCharacter = ruleCharacter; ++ruleNextCharacter;

		if(_isWildcard(ruleCharacter))
		{
			if(ruleNextCharacter != ruleEnd)
			{
				if(*ruleNextCharacter == *begin)
				{
					ruleCharacter = ruleNextCharacter;
					++ruleCharacter;
				}
			}
			
			++begin;
			
			if(begin == end) break;
			
			continue;
		}
		
		if(*ruleCharacter != *begin)
		{
			return false;
		}
		
		++ruleCharacter;
		++begin;
		
		if(begin == end) break;
	}
	
	matchEnd     = begin;
	matchRuleEnd = ruleCharacter;
	
	return true;
}

bool LexerRule::_match(const_iterator& matchEnd,
	const_iterator textBegin, const_iterator textEnd) const
{
	auto ruleEnd = begin();
	
	for(auto ruleCharacter = begin(); ruleCharacter != end(); ++ruleCharacter)
	{
		if(_match(matchEnd, ruleEnd, textBegin, textEnd, ruleCharacter, end()))
		{
			return true;
		}
	}
	
	return false;
}

bool LexerRule::_matchWithEnd(const_iterator begin, const_iterator end) const
{
	std::string::const_reverse_iterator textRbegin(end);
	std::string::const_reverse_iterator textRend(begin);
	
	for(auto ruleCharacter = rbegin(); ruleCharacter != rend(); )
	{
		auto ruleNextCharacter = ruleCharacter; ++ruleNextCharacter;

		if(_isWildcard(ruleCharacter.base()))
		{
			if(ruleNextCharacter != rend())
			{
				if(*ruleNextCharacter == *textRbegin)
				{
					++ruleCharacter;
				}
			}
			
			++textRbegin;
			if(textRbegin == textRend) break;
			continue;
		}
		
		if(*ruleCharacter != *textRbegin)
		{
			return false;
		}
		
		++ruleCharacter;
		++textRbegin;
		
		if(textRbegin == textRend) break;
	}
	
	return true;
}

bool LexerRule::_matchWithBegin(const_iterator textBegin,
	const_iterator textEnd) const
{
	for(auto ruleCharacter = begin(); ruleCharacter != end(); )
	{
		auto ruleNextCharacter = ruleCharacter; ++ruleNextCharacter;

		if(_isWildcard(ruleCharacter))
		{
			if(ruleNextCharacter != end())
			{
				if(*ruleNextCharacter == *textBegin)
				{
					++ruleCharacter;
				}
			}
			
			++textBegin;
			if(textBegin == textEnd) break;
			continue;
		}
		
		if(*ruleCharacter != *textBegin)
		{
			return false;
		}
		
		++ruleCharacter;
		++textBegin;
		
		if(textBegin == textEnd) break;
	}
	
	return true;
}

bool LexerRule::_isWildcard(const_iterator character) const
{
	return _wildcards.count(character) != 0;
}
	

}

}

