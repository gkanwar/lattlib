(* ::Package:: *)
(* Permutation group functionality *)
BeginPackage["QCDLib`sn`"];

(* Unprotect and clear everything. *)
Unprotect["`*"];
ClearAll["`*"];

(* Define everything implicitly by giving usages. *)
YoungTab::usage =
    "YoungTab[asc] = to each key the location in the tableau. \
    there must be no duplicate locations and the keys must be a \
    contiguous range of integers.";
YoungTab::badtype = "The argument `1` is not an Association.";
YoungTab::badkeys = "The keys `1` are not contiguous ints.";
YoungTab::badvalues = "The values `1` are not distinct positions.";
YoungTab::notcanon = "The asc `1` does not satisfy canonical ordering.";
YoungTabx::usage =
    "YoungTabx[keys, vals] = the set of tableaux with keys \
    assigned to valid locations in vals. keys must be a contiguous \
    range of integers and vals unique coordinates. Validity is \
    determined by avoiding decreasing rows or columns.";
YoungTabx::badkeys = "The keys `1` are not contiguous ints.";
YoungTabx::badvalues = "The values `1` are not distinct positions.";
YoungTabx::kvlen = "The keys `1` and values `2` do not have compatible lengths.";
YoungTabx::badshape = "The shape `1` is not valid.";
YoungTabx::badtype = "The arg `1` must be type List.";
PermRep::badkeys = "The keys `1` are not contiguous ints.";
PermRep::badtype = "The arg `1` must be type List.";
toGraphics::usage =
    "toGraphics[obj] = obj in graphical form.";
toGraphics::noconv = "No known graphical form for `1`";
getTabxShape::usage =
    "getTabxShape[tabxVals] = a list of row lengths.
    getTabxShape[tabx] = a list of row lengths.";
regularTabxShapeQ::usage =
    "regularTabxShapeQ[tabxVals] = whether tabx is regular.
    regularTabxShapeQ[tabx] = whether tabx is regular.";
getYamanouchiSymbol::usage =
    "getYamanouchiSymbol[tab] = Yamanouchi symbol of tab.";
getTabxTabList::usage =
    "getTabxTabList[tabx] = a list of YoungTab elements of tabx, in canonical order.";
getTabxRep::usage =
    "getTabxRep[tabx] = a function that returns Sn matrix elements in the tabx \
    representation, given arguments in the range of tabx elements.";
getCSCO::usage =
    "getCSCO[rep] = an array of {C2,C3,...} CSCO as defined by Chen et al.";
getCSCOSubspace::usage =
    "getCSCOSubspace[csco,eigs] = linear subspace of rep with CSCO eigenvalues eigs.";
getCSCOSubspace::badlen = "Length of CSCO `1` doesn't match eigs `2`.";

Begin["`Private`"];

(*** UTILS ***)
consecutiveQ[{}] = True;
consecutiveQ[xs_List] := (xs == Range[Round@Min@xs, Round@Max@xs]);
getTabRow[asc_Association, i_Integer] := Module[{row},
  row = {};
  Map[If[MatchQ[#, (a_ -> {b_, c_} /; b == i)], AppendTo[row, #]]&,
    Normal[asc]];
  SortBy[row, (# /. (a_ -> {b_, c_}) :> c)&]
  ];
getTabCol[asc_Association, i_Integer] := Module[{col},
  col = {};
  Map[If[MatchQ[#, (a_ -> {b_, c_} /; c == i)], AppendTo[col, #]]&,
    Normal[asc]];
  SortBy[col, (# /. (a_ -> {b_, c_}) :> b)&]
  ];
validTabQ[asc_Association] := Module[{rowinds, rows, colinds, cols},
  rowinds = Sort[DeleteDuplicates[Map[#[[1]]&, Values[asc]]]];
  colinds = Sort[DeleteDuplicates[Map[#[[2]]&, Values[asc]]]];
  rows = Map[Keys[getTabRow[asc, #]]&, rowinds];
  cols = Map[Keys[getTabCol[asc, #]]&, colinds];
  And[
    AllTrue[rows, OrderedQ],
    AllTrue[cols, OrderedQ]]
  ];
validTabQ[a___] := (Throw[List[a]]; $Failed);
getTabxRow[vals_List, i_Integer] := Module[{row},
  row = {};
  Map[If[MatchQ[#, {b_,c_} /; b == i], AppendTo[row, #]]&, vals];
  SortBy[row, (# /. {b_,c_} :> c)&]
  ];
getTabxCol[vals_List, i_Integer] := Module[{col},
  col = {};
  Map[If[MatchQ[#, {b_,c_} /; c == i], AppendTo[col, #]]&, vals];
  SortBy[col, (# /. {b_,c_} :> b)&]
  ];
getTabxShape[vals_List] := Module[{rows, maxrow, counts},
  rows = vals[[All,1]];
  maxrow = Max[rows];
  shape = Table[Count[rows, i], {i, maxrow}];
  shape
  ];
getTabxShape[HoldPattern[YoungTabx[keys_,vals_]]] := getTabxShape[vals];

(* Regular tabx must be left justified and decreasing row lengths *)
regularTabxShapeQ[vals_List] := Module[{rows, maxrow, shape},
  maxrow = Max[vals[[All,1]]];
  rows = Table[getTabxRow[vals, i][[All,2]], {i, maxrow}];
  shape = getTabxShape[vals];
  And[
    OrderedQ[Reverse[shape]],
    AllTrue[rows, consecutiveQ]]
  ];
regularTabxShapeQ[YoungTabx[keys_,vals_]] := regularTabxShapeQ[vals];

(* Valid tabx must have consecutive coords in each row and col *)
validTabxShapeQ[vals_List] := Module[{maxrow, rows, maxcol, cols},
  maxrow = Max[vals[[All,1]]];
  maxcol = Max[vals[[All,2]]];
  rows = Table[getTabxRow[vals, i][[All,2]], {i, maxrow}];
  cols = Table[getTabxCol[vals, i][[All,1]], {i, maxcol}];
  And[
    AllTrue[rows, consecutiveQ],
    AllTrue[cols, consecutiveQ]]
  ];

getYamanouchiSymbol[HoldPattern[YoungTab[asc_]]] :=
    Map[asc[#][[1]]&, Reverse[Keys[asc]]];
axialPos[{r_,c_}] := (c-r);
axialDist[coord1_,coord2_] := axialPos[coord2] - axialPos[coord1];

applyPhaseConv[v_List] := Module[{pos, elt},
  pos = Position[v, _?(# != 0 &), 1, 1][[1,1]];
  elt = v[[pos]];
  v * Sign[elt]
  ];

(*** TYPES ***)
YoungTab[asc_Association] \
    /; (If[!consecutiveQ[Keys[asc]],
       Message[YoungTab::badkeys, Keys[asc]]; True, False]) := $Failed;
YoungTab[asc_Association] \
    /; (If[Length@Counts[Values[asc]] != Length@asc,
       Message[YoungTab::badvalues, Values[asc]]; True, False]) := $Failed;
YoungTab[asc_Association] \
    /; (If[Map[Dimensions, Values[asc]] != ConstantArray[{2}, Length@asc],
       Message[YoungTab::badvalues, Values[asc]]; True, False]) := $Failed;
YoungTab[asc_Association] \
    /; (If[!validTabQ[asc],
       Message[YoungTab::notcanon, asc]; True, False]) := $Failed;
YoungTab[asc_] /; (If[!AssociationQ[asc],
  Message[YoungTab::badtype, asc]; True, False]) := $Failed;

YoungTabx[keys_List, vals_List] \
    /; (If[!consecutiveQ[keys],
       Message[YoungTabx::badkeys, keys]; True, False]) := $Failed;
YoungTabx[keys_List, vals_List] \
    /; (If[Length@Counts[vals] != Length@vals,
       Message[YoungTabx::badvalues, vals]; True, False]) := $Failed;
YoungTabx[keys_List, vals_List] \
    /; (If[Length@keys != Length@vals,
       Message[YoungTabx::kvlen, keys, vals]; True, False]) := $Failed;
YoungTabx[keys_List, vals_List] \
    /; (If[!validTabxShapeQ[vals],
       Message[YoungTabx::badshape, vals]; True, False]) := $Failed;
YoungTabx[keys_, vals_] \
    /; (If[!ListQ[keys],
       Message[YoungTabx::badtype, keys]; True,
       If[!ListQ[vals],
         Message[YoungTabx::badtype, vals]; True,
         False]]) := $Failed;

PermRep[keys_List, C_] \
    /; (If[!consecutiveQ[keys],
       Message[PermRep::badkeys, keys]; True, False]) := $Failed;
PermRep[keys_, C_] \
    /; (If[!ListQ[keys],
       Message[PermRep::badtype, keys]; True, False]) := $Failed;

(*** MATH ***)
(* Recursively place smallest remaining elt in a head position *)
getTabxList[HoldPattern[YoungTabx[{n_}, {coord_}]]] := { {n->coord} };
getTabxList[HoldPattern[YoungTabx[keys_,vals_]]] :=
    Module[{minrow, maxrow, heads, sublist, elt, rest},
      (* could be more efficient *)
      minrow = Min[vals[[All,1]]];
      maxrow = Max[vals[[All,1]]];
      mincol = Min[vals[[All,2]]];
      maxcol = Max[vals[[All,2]]];
      heads = DeleteCases[Intersection[
        Table[Module[{r},
          r = getTabxRow[vals, i];
          If[Length@r > 0, First@r, Null]], {i, minrow, maxrow}],
        Table[Module[{c},
          c = getTabxCol[vals, i];
          If[Length@c > 0, First@c, Null]], {i, mincol, maxcol}]],
        Null];
      sublist = {};
      elt = First[keys];
      rest = Rest[keys];
      Table[
        sublist = Join[sublist,
          Map[
            Prepend[#, elt->heads[[i]]]&,
            getTabxList[YoungTabx[rest, DeleteCases[vals, heads[[i]]]]]]],
        {i, Length@heads}];
      sublist
      ];
getTabxTabList[y_YoungTabx] := SortBy[
  Map[YoungTab[Association[#]]&, getTabxList[y]],
  (-getYamanouchiSymbol[#])&];
getSwapAction[i_, HoldPattern[YoungTab[asc_]]] :=
    Module[{p1,p2},
      p1 = asc[i];
      p2 = asc[i+1];
      If[p1[[1]] == p2[[1]],
        {{1, getYamanouchiSymbol[YoungTab[asc]]}},
        If[p1[[2]] == p2[[2]],
          {{-1, getYamanouchiSymbol[YoungTab[asc]]}},
          Module[{sigma, other},
            sigma = axialDist[p1,p2];
            otherAsc = asc;
            otherAsc = AssociateTo[otherAsc, {i->p2, (i+1)->p1}];
            {{1/sigma,
              getYamanouchiSymbol[YoungTab[asc]]},
              {Sqrt[sigma^2 - 1]/Abs[sigma],
                getYamanouchiSymbol[YoungTab[otherAsc]]}}]
          ]]];
(* Form representation on a tabx *)
getTabxRep[HoldPattern[YoungTabx[keys_,vals_]]] :=
    Module[{C, tabs, ysymbs},
      tabs = getTabxTabList[YoungTabx[keys,vals]];
      ysymbs = Map[getYamanouchiSymbol, tabs];
      Table[
        C[i,i+1] = 
          Module[{pos, val, elts, j},
            pos = {};
            val = {};
            For[j = 1, j <= Length@tabs, j++,
              elts = getSwapAction[i, tabs[[j]]];
              Map[(
                AppendTo[val, #[[1]]];
                AppendTo[pos, {FirstPosition[ysymbs, #[[2]]][[1]], j}];
                )&, elts];
              ];
            SparseArray[pos -> val]],
        {i, Min[keys], Max[keys]-1}];
      C[i_,j_] /; (j != i+1) := Module[{cs,is},
        is = Join[Range[i,j-1], Reverse@Range[i,j-2]];
        cs = Map[C[#,#+1]&, is];
        FoldList[Dot, cs][[-1]]];
      PermRep[keys, C]
      ];
(* Form CSCO on a rep *)
getCSCO[HoldPattern[PermRep[keys_,C_]]] :=
    Module[{csco, i, j, curmat},
      csco = {};
      curmat = ConstantArray[0, Dimensions@C[keys[[1]], keys[[2]]]];
      For[j = 2, j <= Length@keys, j++,
        For[i = 1, i < j, i++,
          curmat += C[keys[[i]], keys[[j]]];
          ];
        AppendTo[csco, curmat];
        ];
      csco];
(* Extract CSCO subspace with given eigenvalues *)
getCSCOSubspace[csco_List, eigs_List] \
    /; (If[Length@csco != Length@eigs,
       Message[getCSCOSubspace::badlen, csco, eigs]; True, False]) := $Failed;
getCSCOSubspace[csco_List, eigs_List] :=
    Module[{dim, x, v, sol, sols},
      dim = Dimensions[csco][[2]];
      v = Table[x[i], {i, dim}]; (* vector of vars *)
      sols = {};
      While[True,
        sol = FindInstance[Join[
          Table[csco[[i]] . v == eigs[[i]] v, {i, Length@eigs}],
          {v.v == 1},
          Map[v.# == 0 &, sols]], v];
        If[Length@sol == 0, Break[]];
        AppendTo[sols, v //. sol[[1]]];
        ];
      (* phase convention: first non-zero positive *)
      sols = Map[applyPhaseConv, sols];
      Sort[sols]
      ];

(*** GRAPHICS ***)
Options[makeBox] = {"Size" -> 36};
(* Note: Translating coordinates requires transposing and negating y *)
tabToGraphicsCoord[{b_Integer, c_Integer}] :=
    Module[{x,y},
      x = c-1;
      y = -(b-1);
      {x,y}];
makeBox[{b_Integer, c_Integer}, OptionsPattern[]] :=
    Module[{size, x, y},
      size = OptionValue["Size"];
      {x,y} = tabToGraphicsCoord[{b,c}];
      {LightGray,
        Rectangle[
          Offset[size {x-0.5, y-0.5}],
          Offset[size {x+0.5, y+0.5}]]}
      ];
makeFilledBox[a_Integer -> {b_Integer, c_Integer}, OptionsPattern[{makeBox}]] :=
    Module[{size, x, y},
      size = OptionValue[makeBox, "Size"];
      {x,y} = tabToGraphicsCoord[{b,c}];
      Join[makeBox[{b,c}],
        {Text[Style[a, 0.5 size, Black, Bold], Offset[size {x,y}]]}]
      ];
tabBounds[coords_] := Module[{xs, ys, size},
  xs = coords[[All,2]];
  ys = -coords[[All,1]];
  size = OptionValue[makeBox, Options[makeBox], "Size"];
  size {{Min@xs - 1.5, Max@xs - 0.5}, {Min@ys + 0.5, Max@ys + 1.5}}
  ];
toGraphics[HoldPattern[YoungTab[asc_]]] := Module[{bounds, sizes, gs},
  bounds = tabBounds[Values[asc]];
  sizes = bounds[[All,2]] - bounds[[All,1]];
  gs = Join@@Map[makeFilledBox, Normal[asc]];
  Show[Graphics[gs, PlotRange->bounds], ImageSize->sizes]
  ];
toGraphics[HoldPattern[YoungTabx[keys_,vals_]]] := Module[{bounds, sizes, gs},
  bounds = tabBounds[vals];
  sizes = bounds[[All,2]] - bounds[[All,1]];
  center = Map[Mean, bounds];
  gs = Join@@Map[makeBox, vals];
  Show[{
    Graphics[gs, PlotRange->bounds],
    Graphics[Text[Style[ToString[keys], Black, Bold], Offset[center]]]
    }, ImageSize->sizes]
  ];
toGraphics[a___] := Message[toGraphics::noconv, List[a]];

End[]; (* `Private` *)

(* Log and protect everything exported. *)
Print["Loaded " <> Context[] <> " with symbols: ", Names["`*"]];
Protect["`*"];

EndPackage[];