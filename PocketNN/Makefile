# target: dependencies
# 	action

CXX = g++
CXXFLAGS = -std=c++11

output: main.o pktnn_examples.o pktnn_fc.o pktnn_mat.o pktnn_tools.o pktnn_loss.o pktnn_loader.o pktnn_layer.o pktnn_actv.o pktnn_mat3d.o
	$(CXX) $(CXXFLAGS) main.o pktnn_examples.o pktnn_fc.o pktnn_mat.o pktnn_tools.o pktnn_loss.o pktnn_loader.o pktnn_layer.o pktnn_actv.o pktnn_mat3d.o -o output

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp

pktnn_examples.o: pktnn_examples.cpp pktnn_examples.h
	$(CXX) $(CXXFLAGS) -c pktnn_examples.cpp

pktnn_fc.o: pktnn_fc.cpp pktnn_fc.h
	$(CXX) $(CXXFLAGS) -c pktnn_fc.cpp

pktnn_mat.o: pktnn_mat.cpp pktnn_mat.h
	$(CXX) $(CXXFLAGS) -c pktnn_mat.cpp

pktnn_tools.o: pktnn_tools.cpp pktnn_tools.h
	$(CXX) $(CXXFLAGS) -c pktnn_tools.cpp

pktnn_loss.o: pktnn_loss.cpp pktnn_loss.h
	$(CXX) $(CXXFLAGS) -c pktnn_loss.cpp

pktnn_loader.o: pktnn_loader.cpp pktnn_loader.h
	$(CXX) $(CXXFLAGS) -c pktnn_loader.cpp

pktnn_layer.o: pktnn_layer.cpp pktnn_layer.h
	$(CXX) $(CXXFLAGS) -c pktnn_layer.cpp

pktnn_actv.o: pktnn_actv.cpp pktnn_actv.h
	$(CXX) $(CXXFLAGS) -c pktnn_actv.cpp

pktnn_mat3d.o: pktnn_mat3d.cpp pktnn_mat3d.h
	$(CXX) $(CXXFLAGS) -c pktnn_mat3d.cpp

clean:
	rm *.o output output.txt



