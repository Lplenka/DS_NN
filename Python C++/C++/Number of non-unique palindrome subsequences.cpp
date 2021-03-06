/*
For every string given as input, you need to tell us the number of subsequences of 
it that are palindromes (need not necessarily be distinct). 
Note that the empty string is not a palindrome.

*/

#include<iostream>
#include<string.h>
using namespace std;
 
int a[52][52];
 
int count(int low, int high, char str[])
{
	if(low>high)
	 return 0;
	 
	if(low==high)
	 return 1;
	
	if(a[low][high]!=-1)
	 return a[low][high];
	else{
	a[low+1][high] = count(low+1,high,str);
	a[low][high-1]= count(low,high-1,str);
	}
	int p = a[low+1][high] + a[low][high-1];
	
	if(str[low]!=str[high])
	 return  p- count(low+1,high-1,str);
	else
	 return p +1;
}
 
int main()
{
	int test;
	cin>>test;
	while(test--){
	char str[52];
	cin>>str;
	int len=strlen(str);
	for(int i=0;i<=51;i++)
	{
		for(int j=0;j<=51;j++)
		 a[i][j]=-1;
	}
	cout<<count(0,len-1,str)<<endl;
	}
	return 0;
}
