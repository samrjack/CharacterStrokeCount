#!c:\perl\bin\perl.exe
#strokecounter.pl
#Counts character strokes
#Author: Stephen Politzer-Ahles
#Last modified: 2012-03-05
#You are free to use or modify this code as you see fit, but please credit accordingly and do not sell it in any way

use Encode;                                   				#Modules

open( totalstrokes, "< totalstrokes.txt" ) or die( "Something broke");	#Open the data (a table with the stroke count for every Unihan character)

while( chomp( $line = <totalstrokes> ) ){				#Populate %strokecount hash (the same table from above, in hash format)
	($c, $s) = split(/\t/, $line);
	$strokecount{ $c } = $s;
}
close totalstrokes;							#That file is not needed anymore

$linenumber = 0;
open datatocount, "< datatocount.txt" or die( "Something broke");
while( chomp( $line = <datatocount> ) ){				#Read the words/characters that we want stroke counts for into the array @characterstocount
	$characterstocount[$linenumber] = $line;
	$linenumber = $linenumber + 1;
}


open strokecounts, "> strokecounts.txt";				#Open the file that we will write the stroke counts to
foreach $character ( @characterstocount ) {				#Iterate through the characters that we want to count
	$decodedstr = decode( "utf8", $character);			#Decode characters from text into binary

	$perchar_strokes = "";						#Set up variables to keep track of the stroke counts as we get them. $perchar_strokes
	$total_strokes = 0;						#  is a string with the stroke count of each character, and $total_strokes is an int
									#  with the total stroke count for the word
		
	while( $x = chop( $decodedstr ) ){                         	#Take one character at a time off the end of the string
		$hex = dec_to_hex( ord($x) );				#Turn the Unicode decimal into hex (see dec_to_hex() function below)
									# $strokecount{ $hex } is the stroke count for this character (from the %strokecount hash)
		$total_strokes += $strokecount{ $hex };			#Increment the total strokes by the stroke count of the current character
									#Add the stroke count for this character to the string of $perchar_strokes
		$perchar_strokes = $strokecount{ $hex } . " " . $perchar_strokes;
	}
	print( strokecounts $total_strokes ."\n");			#Print the total strokes for this word out to the file
									#If we want strokes per character rather than total strokes for the word, we could 
									#  probably print $perchar_strokes instead of $total_strokes. I haven't tried it
}



sub hex_to_dec{
		#Function documentation
		#Takes a hex number and turns it into a decimal number, which is returned

		$hex = shift();						#Get the argument
		$dec = 0;						#Initialize the $dec variable to 0	

		$pos = 0;						#The place (start at ones place, then go up to 10s, 100s, etc.)
		
		while( $hex ){		
  			$digit = numberify( chop( $hex ) );		#Get the last hex digit from the number and convert it to decimal
			$dec += $digit*( 16**$pos );			#Add that to $dec
			$pos += 1;					#Go up to the next place
		}
		
		return $dec;
}

sub dec_to_hex{
		#Function documentation
		#Takes a decimal number and turns it into a decimal number, which is returned

		$dec = shift();						#Get the argument
		$hex = '';						#Initialize the $hex variable to an empty string
		
		while( $dec ){
			$digit = hexnumberify( $dec % 16 );		#Get the last bit of the decimal (using modulus division) and convert it into a hex digit
			$hex = $digit . $hex;				#Add the digit to $hex
			$dec = int( $dec/16 );				#Go up to the next place of the decimal (by removing the part from the end that was just converted to hex)
		}

		return $hex;
}

sub numberify{
		#Function documentation
		#Takes a single hex digit in string format (from 1 to F) and converts it to a decimal number
    		$x = shift();						#Get the argument
		
		if( $x eq "1" ){ $x = 1; }				#Take the hex digit and convert it to a decimal number
		elsif( $x eq "2" ){ $x = 2; }
		elsif( $x eq "3" ){ $x = 3; }
		elsif( $x eq "4" ){ $x = 4; }
		elsif( $x eq "5" ){ $x = 5; }
		elsif( $x eq "6" ){ $x = 6; }
		elsif( $x eq "7" ){ $x = 7; }
		elsif( $x eq "8" ){ $x = 8; }
		elsif( $x eq "9" ){ $x = 9; }
		elsif( $x eq "A" ){ $x = 10; }
		elsif( $x eq "B" ){ $x = 11; }
		elsif( $x eq "C" ){ $x = 12; }
		elsif( $x eq "D" ){ $x = 13; }
		elsif( $x eq "E" ){ $x = 14; }
		elsif( $x eq "F" ){ $x = 15; }
		
		return $x
}

sub hexnumberify{
		#Function documentation
		#Takes a decimal number below 16 in number format and converts it to a hex digit
    		$x = shift();						#Get the argument
		
		if( $x < 10 ){ }					#If the decimal is below 10, do nothing
		elsif( $x == 10 ){ $x = 'A'; }				#If the decimal is above 10, turn it into one of the hex letters
		elsif( $x == 11 ){ $x = 'B'; }
		elsif( $x == 12 ){ $x = 'C'; }
		elsif( $x == 13 ){ $x = 'D'; }
		elsif( $x == 14 ){ $x = 'E'; }
		elsif( $x == 15 ){ $x = 'F'; }
		
		return $x
}

exit;