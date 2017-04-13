#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>

using namespace std;

int main() {
    string input;
    while (getline(cin, input)) {
        if (input[0] == 'n') {
            istringstream iss(input);
            vector <string> tokens{istream_iterator < string > {iss}, istream_iterator < string > {}};
			cout << "allNodes.add(new Node(" << tokens[1] << ", " << tokens[3] << ", " << tokens[4] << "));\n";
        } else if (input[0] == 'g' && input[3] == 'e') {
            istringstream iss(input);
            vector <string> tokens{istream_iterator < string > {iss}, istream_iterator < string > {}};
            if (tokens[8][0] == '1')
				cout << "new Link(" << tokens[2] << ", " << tokens[3] << ", " << tokens[4] << ", allNodes);\n";
        }
    }
    return 0;
}
