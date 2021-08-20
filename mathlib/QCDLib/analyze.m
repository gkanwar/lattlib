(* ::Package:: *)
(* Data analysis functionality *)
BeginPackage["QCDLib`analyze`"];

(* Unprotect and clear everything. *)
Unprotect["`*"];
ClearAll["`*"];

(* Define everything implicitly by giving usages. *)
bootstrap::usage =
    "bootstrap[series, Nboot, f] = bootstrap estimate of sample \
    error of f on the series.";
autocorrs::usage =
    "autocorrs[series, maxt] = evaluate samples of rho(t) for t in [0,maxt].";
stdDev::usage =
    "stdDev[l] = standard dev of numerical ensemble, or zeros if length 1.";
constantFit::usage =
    "constantFit[means, covars] = best fit constant, error, and chi squared.";
constantSysFit::usage =
    "constantSysFit[means, covars, ti, tf] = best fit constant, stat + syst \
    error and chi squared given a window [ti,tf].";
getMeffs::usage =
    "getMeffs[corrs] = the meffs and errors of a given ensemble \
    of correlators with dimensions Nmeas x Lt";
meff::usage =
    "meff[corrs] = compute effective masses given time-sequence of correlators.";
meffAcosh::usage =
    "meffAcosh[corrs] = compute acosh effective masses given correlators.";

Begin["`Private`"];
(* Bootstrapping error analysis *)
Options[bootstrap] = {"BinSize"->1, "RawBoot"->False, "Progress"->False};
bootstrap[series_List, Nboot_Integer, f_, OptionsPattern[]] :=
    Module[{Nmeas, binSize, rawBoot, progress, binSeries, bootInds, bi, boots},
      Nmeas = Length[series];
      binSize = OptionValue["BinSize"];
      rawBoot = OptionValue["RawBoot"];
      progress = OptionValue["Progress"];
      (* Assert[Mod[Nmeas, binSize] == 0]; *)
      (* Binning *)
      If[binSize != 1,
        binSeries = Map[Mean, Partition[series, UpTo[binSize]]],
        binSeries = series];
      Nmeas = Length[binSeries];
      boots = {};
      If[progress, PrintTemporary[
        ProgressIndicator[Dynamic[bi], {1, Nboot+1}]]];
      For[bi = 1, bi <= Nboot, bi++,
        bootInds = RandomInteger[{1,Nmeas}, Nmeas];
        AppendTo[boots, f[binSeries[[bootInds]]]];
        ];
      If[rawBoot, boots,
        {2 f[binSeries] - Mean[boots], StandardDeviation[boots]}]
      ];

(* (normalized) autocorr samples, which can be fed into bootstrap mean
   for errors *)
autocorrs[series_List, maxt_Integer] :=
    Module[{mean = Mean@series, out},
      out = Table[(series[[;;-t]] - mean) (series[[t;;]] - mean), {t, 1, maxt}];
      out / Mean[out[[1]]]
      ];

(* Custom StdDev that handles single element lists *)
stdDev[{element_}] := ConstantArray[0, Dimensions@element];
stdDev[l_List] := StandardDeviation[l];

(* Constant value fitter, h/t mlwagman *)
constantFit[means_, covars_] :=
    Module[{length, M, MSolver, MInv1, H, Delta, deltam, m, chiSq, dof, chiSqDof},
      length = Length@means;
      M = 1/2 (covars + ConjugateTranspose[covars]);
      MSolver = LinearSolve[M, Method->"Cholesky"];
      MInv1 = MSolver[ConstantArray[1,length]];
      If[Not@MatchQ[MInv1, _List], Throw["Ill-conditioned covariance matrix!"]];
      H = 2 ConstantArray[1,length].MInv1;
      Delta = 2 / H;
      deltam = Sqrt[Delta];
      m = Delta means . MInv1;
      chiSq = (means - m) . MSolver[means - m];
      dof = length - 1;
      chiSqDof = chiSq/dof;
      {m, deltam, chiSqDof}];

(* Constant value fitter with syst estimate, h/t mlwagman *)
constantSysFit[means_, covars_, ti_, tf_] :=
    Module[{
      m1, \[Delta]m1, \[Chi]sq1, m2, \[Delta]m2, \[Chi]sq2,
      m3, \[Delta]m3, \[Chi]sq3, \[Delta]msys},
      
      {m1,\[Delta]m1,\[Chi]sq1} = constantFit[
        means[[ti;;tf]], covars[[ti;;tf,ti;;tf]]];
      {m2,\[Delta]m2,\[Chi]sq2} = constantFit[
        means[[ti+1;;tf+1]], covars[[ti+1;;tf+1,ti+1;;tf+1]]];
      {m3,\[Delta]m3,\[Chi]sq3} = constantFit[
        means[[ti+2;;tf+2]], covars[[ti+2;;tf+2,ti+2;;tf+2]]];
      \[Delta]msys = 1/2 (Max[m1,m2,m3] - Min[m1,m2,m3]);
      
      {m2, \[Delta]m2, \[Delta]msys, \[Chi]sq2, {ti,tf}}];

(* Effective masses from correlator data *)
Options[getMeffs] = {"Nboot" -> 8, "BinSize" -> 1, "RawBoot" -> False};
getMeffs[corrs_, OptionsPattern[]] :=
    Module[{
      Nmeas, Lt, Nboot, binSize, rawBoot, binNmeas, binCorrs,
      bi, bootInds, meffs, bootCorrs, bootMeffs},
      Nboot = OptionValue["Nboot"];
      binSize = OptionValue["BinSize"];
      rawBoot = OptionValue["RawBoot"];
      {Nmeas, Lt} = Dimensions@corrs;
      (* Binning *)
      Assert[Mod[Nmeas, binSize] == 0];
      binNmeas = Nmeas / binSize;
      binCorrs = Map[Mean, Partition[corrs, binSize]];
      (* Bootstrap *)
      meffs = {};
      For[bi = 1, bi <= Nboot, bi++,
        bootInds = RandomInteger[{1,binNmeas}, binNmeas];
        bootCorrs = Re@Mean@binCorrs[[bootInds]];
        bootMeffs = Log@bootCorrs[[;;-2]] - Log@bootCorrs[[2;;]];
        AppendTo[meffs, bootMeffs];
        ];
      If[rawBoot,
        meffs,
        {Mean@meffs, stdDev@meffs}]];

(* Raw meff calculation on time-series *)
meff[corrs_] := Module[{meanCorr},
  meanCorr = Mean@corrs;
  Re[Log[meanCorr[[;; -2]]] - Log[meanCorr[[2 ;;]]]]
  ];

meffAcosh[corrs_] := Module[{meanCorr},
  meanCorr = Mean@corrs;
  ArcCosh[(meanCorr[[;; -3]] + meanCorr[[3 ;;]]) / (2 meanCorr[[2;;-2]])]
  ];

End[]; (* `Private` *)

(* Log and protect everything exported. *)
Print["Loaded " <> Context[] <> " with symbols: ", Names["`*"]];
Protect["`*"];

EndPackage[];