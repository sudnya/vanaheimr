/*! \file   LexerRule.h
	\date   April 28, 2013
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the LexerRule class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <set>

namespace vanaheimr
{

namespace parser
{

/* \brief A class for representing a regular expression used to match a
		Lexer token
*/
class LexerRule
{
public:
	explicit LexerRule(const std::string& regex);
		
public:
	bool canMatchWithBegin(const std::string&) const;
	bool canMatchWithEnd(const std::string&) const;
	bool canOnlyMatchWithBegin(const std::string&) const;
	bool canOnlyMatchWithEnd(const std::string&) const;
	bool canMatch(const std::string&) const;
	bool isExactMatch(const std::string&) const;

public:
	const std::string& toString() const;

public:
	typedef std::string::iterator       iterator;
	typedef std::string::const_iterator const_iterator;
	
	typedef std::string::reverse_iterator       reverse_iterator;
	typedef std::string::const_reverse_iterator const_reverse_iterator;

public:
	      iterator begin();
	const_iterator begin() const;

	      iterator end();
	const_iterator end() const;
	
	      reverse_iterator rbegin();
	const_reverse_iterator rbegin() const;

	      reverse_iterator rend();
	const_reverse_iterator rend() const;
	
public:
	bool   empty() const;
	size_t  size() const;

private:
	bool _match(const_iterator& matchEnd, const_iterator& matchRuleEnd,
		const_iterator begin, const_iterator end,
		const_iterator ruleBegin, const_iterator ruleEnd) const;
	bool _match(const_iterator& matchEnd, const_iterator begin,
		const_iterator end) const;
	bool _isExactMatch(const std::string& text) const;
	bool _matchWithEnd(const_iterator begin, const_iterator end) const;
	bool _matchWithBegin(const_iterator begin, const_iterator end) const;
	bool _canMatchWithBegin(const std::string& text) const;
	bool _canMatchWithEnd(const std::string& text) const;
	bool _canMatch(const std::string&) const;
	
	bool _isWildcard(const_iterator) const;
	
private:
	typedef std::set<const_iterator> IteratorSet;
	
private:
	std::string _regex;
	IteratorSet _wildcards;
	

};

}

}

