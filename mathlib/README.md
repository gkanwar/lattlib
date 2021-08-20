QCDLib Mathematica Paclet
=========================
Contains various modules that I have found useful at some point in my grad school experience. The simplest approach to loading the library is via the online paclet by including the following initialization cell in your notebook

```
If[MatchQ[PacletInformation["QCDLib"], {}],
  URLSave[
   "https://scripts.mit.edu/~gurtej/mma_paclets/qcdlib.paclet", 
   "/tmp/qcdlib.paclet"];
  Print@PacletInstall["/tmp/qcdlib.paclet"];
  Print["Installed: ", PacletInformation["QCDLib"]];
  ];
Needs["QCDLib`"];
```

Alternatively, build the paclet using the Makefile and run the `PacletInstall` command once with the appropriate path.