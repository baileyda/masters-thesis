#include <stdio.h>
#include <assert.h>

#include <cufft.h>

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace std;

typedef unsigned int uint;
typedef cufftComplex Complex;

extern "C" void loadData(char* real, char* imaginary, uint size, Complex* pData);

extern "C" void loadFilter(char* filter, uint size, Complex* pData);

extern "C" void writeData(Complex* pResult, char* output, uint size);

extern "C" void writeMatrixData(float* pResult, char* output, int num, long Np, long P);

extern "C" void writeMatrixDataComplex(Complex* pResult, char* output, int num, long Np, long P);

void split(const string& str, vector<string>& splitStr, string delimiter);

void loadData(char* real, char* imaginary, uint size, Complex* pData) {

	//open the input file and read in the desired data
	ifstream realFile(real);
	ifstream imagFile(imaginary);

	string instring;
	vector<string> splitStrReal;
	vector<string> splitStrImag;
	uint index = 0;
	uint numRows = 0;

	//ensure we were able to successfully open the input file for reading
	if(realFile.is_open() && imagFile.is_open()) {
		printf("properly opened the given signal data files\n");
		//read data until we reach the end of the file
		while(!realFile.eof() && !imagFile.eof()) {
			//get a full line out of the file
			getline(realFile, instring);
			splitStrReal.clear();
			//split the string up into individual elements
			split(instring, splitStrReal, "\t");

			//get a full line out of the file
			getline(imagFile, instring);
			splitStrImag.clear();
			//split the string up into individual elements
			split(instring, splitStrImag, "\t");

			assert(splitStrReal.size() == splitStrImag.size());

			//check that the size of this row is not larger than the defined size
			if(splitStrReal.size() > size) {
				//if it is throw and error and terminate the program.
				cout << "ERROR: row " << numRows << "'s size was larger than the defined size of " << size << "\n";
				delete pData;
				realFile.close();
				imagFile.close();
				exit(0);
			}
			
			//if the row is properly sized then we will store the data
			for(uint q = 0; q < splitStrReal.size(); q++) {
				//check that our index is not about to address out
				//of the bounds of our defined data pointer.
				if(index < size) {
					//convert the string into a float
					pData[index].x = (float)atof(splitStrReal[q].c_str());
					pData[index].y = (float)atof(splitStrImag[q].c_str());

					//increment our running index counter
					index++;
				}
				else {
					//if the index is out of bounds then throw an
					//error and terminate the program.
					cout << "ERROR: read in more data than the defined size specified!!\n";
					delete pData;
					realFile.close();
					imagFile.close();
					exit(0);
				}
			}
			//increment the number of rows read in
			numRows++;
		}

		realFile.close();
		imagFile.close();
	}
	else {
		cout << "ERROR: could not open file " << real << " for reading\n";
	}
}


void loadFilter(char* filter, uint size, Complex* pData) {

	//open the input file and read in the desired data
	ifstream inFile(filter);

	string instring;
	vector<string> splitStr;
	uint index = 0;
	uint numRows = 0;

	//ensure we were able to successfully open the input file for reading
	if(inFile.is_open()) {
		printf("properly opened the given filter data file\n");
		//read data until we reach the end of the file
		while(!inFile.eof()) {
			//get a full line out of the file
			getline(inFile, instring);
			splitStr.clear();
			//split the string up into individual elements
			split(instring, splitStr, "\t");

			//check that the size of this row is not larger than the defined size
			if(splitStr.size() > size) {
				//if it is throw and error and terminate the program.
				cout << "ERROR: row " << numRows << "'s size was larger than the defined size of " << size << "\n";
				delete pData;
				inFile.close();
				exit(0);
			}
			
			//if the row is properly sized then we will store the data
			for(uint q = 0; q < splitStr.size(); q++) {
				//check that our index is not about to address out
				//of the bounds of our defined data pointer.
				if(index < size) {
					//convert the string into a float
					pData[index].x = (float)atof(splitStr[q].c_str());
					pData[index].y = (float)atof(splitStr[q].c_str());

					//increment our running index counter
					index++;
				}
				else {
					//if the index is out of bounds then throw an
					//error and terminate the program.
					cout << "ERROR: read in more data than the defined size specified!!\n";
					delete pData;
					inFile.close();
					exit(0);
				}
			}
			//increment the number of rows read in
			numRows++;
		}

		inFile.close();
	}
	else {
		cout << "ERROR: could not open file " << filter << " for reading\n";
	}
}


void writeData(Complex* pResult, char* output, uint size) {
	//open the given file for output
	string real;
	real.assign(output);
	real.append("_real.txt");
	string imag;
	imag.assign(output);
	imag.append("_imag.txt");
	ofstream outReal(real.c_str());
	ofstream outImag(imag.c_str());

	if(outReal.is_open() && outImag.is_open()) {
		//loop over all the resulting data and write it to the file
		for(uint q = 0; q < size; q++) {
			outReal << pResult[q].x << "\t";
			outImag << pResult[q].y << "\t";
		}

		outReal.close();
		outImag.close();
	}
	else {
		cout <<"ERROR: could not open " << output << " for output\n";
		return;
	}
}

void writeMatrixData(float* pResult, char* output, int num, long Np, long P)
{
	cout << "Writing matrix data to a file...\n";
	//open the given file for output
	/*
	string real;
	real.assign(output);
	real.append("_real.txt");
	string imag;
	imag.assign(output);
	imag.append("_imag.txt");
	ofstream outReal(real.c_str());
	ofstream outImag(imag.c_str());
	*/

	char buffer[33];
	string temp;
	temp.assign(output);
	temp.append("_");
	_itoa(num, buffer, 10);
	temp.append(buffer);
	temp.append(".txt");
	ofstream out(temp.c_str());

	//if(outReal.is_open() && outImag.is_open()) {
	if(out.is_open()) {
		//loop over all the resulting data and write it to the file
		for(long q = 0; q < Np*P; q++) {
			if(q % Np == 0) {
				out << "\n";
				//outReal << "\n";
				//outImag << "\n";
			}
			out << pResult[q] << "\t";
			//outReal << pResult[q].x << "\t";
			//outImag << pResult[q].y << "\t";
		}
		out.close();
		//outReal.close();
		//outImag.close();
	}
	else {
		cout <<"ERROR: could not open " << output << " for output\n";
		return;
	}
}

void writeMatrixDataComplex(Complex* pResult, char* output, int num, long Np, long P)
{
	cout << "Writing matrix data to a file...\n";
	//open the given file for output
	char buffer[33];

	string real;
	real.assign(output);
	real.append("_real_");
	_itoa(num, buffer, 10);
	real.append(buffer);
	real.append(".txt");
	string imag;
	imag.assign(output);
	imag.append("_imag_");
	_itoa(num, buffer, 10);
	imag.append(buffer);
	imag.append(".txt");
	ofstream outReal(real.c_str());
	ofstream outImag(imag.c_str());
	

	if(outReal.is_open() && outImag.is_open()) {
		//loop over all the resulting data and write it to the file
		for(long q = 0; q < Np*P; q++) {
			if(q % Np == 0) {
				outReal << "\n";
				outImag << "\n";
			}
			outReal << pResult[q].x << "\t";
			outImag << pResult[q].y << "\t";
		}
		outReal.close();
		outImag.close();
	}
	else {
		cout <<"ERROR: could not open " << output << " for output\n";
		return;
	}
}


void split(const string& str, vector<string>& splitStr, string delimiter)
{
	string::size_type lastPos = 0;
	// Find first delimiter.
    string::size_type pos = str.find(delimiter, lastPos);

	while (pos != string::npos || lastPos != string::npos)
    {
		if(pos == string::npos)
		{
			splitStr.push_back(str.substr(lastPos, str.size() - lastPos));

			lastPos = string::npos;
		}
		else
		{
			if(str.substr(lastPos, pos - lastPos) != "")
				// Found a token, add it to the vector.
				splitStr.push_back(str.substr(lastPos, pos - lastPos));
	        
			// set previous position = to delimiters position + 1
			lastPos = pos + 1;
	        
			// Find next delimiter
			pos = str.find(delimiter, lastPos);
		}
    }
}


void usage() {
	cout << "gpu_signal_processing.exe -signal sig.txt -filter fil.txt -output out.txt -s int -f int [-t float -d int]\n";
	cout << "-signal: specifies the file containing input signal data\n";
	cout << "-filter: specifies the file containing input filter data\n";
	cout << "-output: specifies the file to write the results out to\n";
	cout << "-s [val]: specifies the signal size\n";
	cout << "-f [val]: specifies the filter size\n";
	cout << "-t [0-1): [optional] turns on thresholding of the signal, value given is percent of max to threshold by\n";
	cout << "-d [val]: [optional] turns on signal decimation, value specifes new size of the signal\n";
}