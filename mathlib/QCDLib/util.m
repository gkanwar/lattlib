(* ::Package:: *)
(* Abstract utilities *)
BeginPackage["QCDLib`util`"];

(* Unprotect and clear everything. *)
Unprotect["`*"];
ClearAll["`*"];

(* Define everything implicitly by giving usages. *)
(* CACHING *)
cacheable::usage =
    "cacheable[array] = True if purely numeric array.";
cache::usage =
    "cache[f][arg1, arg2, ...] = cache the output from f acting on the args \
    for fast return in future executions.";
(* MATH *)
ones::usage =
    "ones[l] = constant array of ones of length `l`.";
coordMod::usage =
    "coordMod[coord, dims] = ND coord modulo dimensions.";
addQuad::usage =
    "addQuad[a, b, ...] = adds errors a, b, etc in quadrature.";
symmMat::usage =
    "symmMat[M] = (1/2)(M + M^T).";
nanToZero::usage =
    "nanToZero[n] = flush n to zero if n is NaN.";
comm::usage =
    "comm[m1,m2] = m1.m2 - m2.m1";
wrap::usage =
    "wrap[p] = p - 2Pi nu such that -Pi < p < Pi.";
(* META-PROGRAMMING *)
makeEvalSupp::usage =
    "makeEvalSupp[bool] = function that holds arguments if bool is True, \
    otherwise the Identity.";
deepEq::usage =
    "deepEq[f1, f2] = assign f2 into f1 including all attributes.";
getOptsFor::usage =
    "getOptsFor[fn, opts] = give list of options in opts for fn.";
symbolsWithDownValues::usage =
    "symbolsWithDownValues[context] = list all symbols with upvalues in context.";
dumpGlobalFns::usage =
    "dumpGlobalFns[] = dump info on all global functions.";
(* FILES *)
readRel::usage =
    "readRel[relFName, mode] = reads relative file to object specified by mode.";
writeRel::usage =
    "writeRel[relFName, obj, mode] = writes relative object to file with specified mode.";
exportRel::usage =
    "exportRel[relFName, obj] = exports obj to file relative to NB dir.";
relPath::usage =
    "relPath[relFname] = relative path to NB dir.";
arrayReshape::usage =
    "arrayReshape[arr, shape] = reshapes into shape checking that the sizes match.";

(* Exported constants *)
(* COLORS *)
mmaColors = ColorData[97, "ColorList"];
(* MARKERS *)
mmaPlotMarkers = Graphics`PlotMarkers[];
(* CACHING *)
CacheDir = Module[{fname},
  fname = Quiet[NotebookFileName[]];
  If[MatchQ[fname, $Failed], $Failed, fname <> ".mmaCache"]
  ];


Begin["`Private`"];

(*** CACHING ***)
(* Cache location *)
If[!MatchQ[CacheDir, $Failed],
  If[!DirectoryQ[CacheDir], CreateDirectory[CacheDir]];
  (* Only numeric n-dim arrays are cacheable *)
  cacheable[expr_?ArrayQ] := And@@Map[NumericQ, Flatten@expr];
  cacheable[___] = False;
  (* Cache fn *)
  ClearAll[cache];
  Options[cache] = {"Refresh" -> False};
  cache[f_, OptionsPattern[]][args___] := Module[{s, clearCache, cacheIndex, cacheFile, list, ndims, dims, out},
    clearCache = OptionValue["Refresh"];
    s = SymbolName[f] <> "@" <> ToString[List[args]];
    cacheIndex = ToString@Hash[s];
    cacheFile = CacheDir <> "/" <> SymbolName[f] <> "." <> cacheIndex <> ".dat";
    If[!FileExistsQ[cacheFile] || clearCache,
      (* Cache miss / refresh *)
      If[clearCache,
        PrintTemporary["Cache refresh (", cacheIndex, "): ", (*s*) SymbolName[f]],
        PrintTemporary["Cache miss (", cacheIndex, "): ", (*s*) SymbolName[f]]];
      out = f[args];
      Assert[cacheable@out];
      dims = Dimensions@out;
      ndims = Length@dims;
      list = Join[{ndims}, dims, Flatten@out];
      BinaryWrite[cacheFile, list, "Real64"];
      Close[cacheFile]
      ];
    (* Always reload from cache *)
    PrintTemporary["Cache load (", cacheIndex, "): ", (*s*) SymbolName[f]];
    list = BinaryReadList[cacheFile, "Real64"];
    ndims = Round@Re[list[[1]]];
    dims = Round@Re[list[[1+1;;1+ndims]]];
    out = ArrayReshape[list[[2+ndims;;]], dims];
    out]
  ]

(*** MATH ***)
ones[l_] := ConstantArray[1, l];
coordMod[c_, dims_] :=
    Mod[c - ones[Length[dims]], dims] + ones[Length[dims]];
addQuad[a___] := Sqrt[Total@Map[#^2 &, List[a]]];
symmMat[M_] := (1/2)(M + Transpose[M]);
nanToZero[n_ ?NumberQ] := n;
nanToZero[a_] := 0.;
comm[m1_,m2_] := (m1.m2 - m2.m1);
wrap[p_] := Mod[p+Pi, 2 Pi]-Pi;


(*** META-PROGRAMMING ***)
(* Handy eval suppressor metafn *)
makeEvalSupp[b_] := If[b, Identity,
    Module[{evalSupp},
           SetAttributes[evalSupp, HoldAll];
           evalSupp[f] := Null;
           evalSupp]];
(* Deep assign metafn, to copy over attributes *)
deepEq[f1_, f2_] := Scan[(#[f1] = #[f2];)&, {Attributes}];
getOptsFor[fn_, opts___] := FilterRules[List[opts], Options[fn]];
symbolsWithDownValues[context_] := Select[
  Names[context<>"*"], (
    NameQ[ToString[Symbol[#]]] &&
    Not@MemberQ[Attributes[#], Temporary] &&
    Length@DownValues[Evaluate[Symbol[#]]] > 0
    )&];
dumpGlobalFns[] := Map[Information, symbolsWithDownValues["Global`"]];

(*** FILES ***)
(* Relative path read *)
readRel[fname_, mode_] :=
    BinaryReadList[NotebookDirectory[] <> "/" <> fname, mode];
(* Relative path binary write *)
writeRel[fname_, obj_, mode_] :=
    (BinaryWrite[NotebookDirectory[] <> "/" <> fname, obj, mode];
      Close[NotebookDirectory[] <> "/" <> fname]);
(* Relative path export *)
exportRel[relFName_, obj_] := Export[NotebookDirectory[] <> "/" <> relFName, obj];
(* Relative path *)
relPath[relFName_] := NotebookDirectory[] <> "/" <> relFName;
(* Strong array reshape *)
arrayReshape[arr_, shape_] := (
  Assert[Length@Flatten@arr == FoldList[Times, 1, shape][[-1]]];
  ArrayReshape[arr, shape]);

End[]; (* `Private` *)

(* Log and protect everything exported. *)
Print["Loaded " <> Context[] <> " with symbols: ", Names["`*"]];
Protect["`*"];

EndPackage[];