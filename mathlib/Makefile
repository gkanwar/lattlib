APPNAME=QCDLib

all: qcdlib.paclet

clean:
	rm qcdlib.paclet
qcdlib.paclet: $(APPNAME)/PacletInfo.m $(APPNAME)/Kernel/*.m
	zip -r $@ $(APPNAME)
deploy: qcdlib.paclet
	scp qcdlib.paclet gurtej@athena.dialup.mit.edu:web_scripts/mma_paclets/
