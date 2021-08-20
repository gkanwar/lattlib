(* ::Package:: *)
(* Any functionality related to visualization. This is distinct from plotting
   and data analysis functionality. *)
BeginPackage["QCDLib`visualize`", {"QCDLib`plot`"}];

(* Unprotect and clear everything. *)
Unprotect["`*"];
ClearAll["`*"];

(* Define everything implicitly by giving usages. *)
drawpath::usage =
    "drawpath[dirSeq] = drawing of the path consisting of the given \
    direction sequence.";
draw2DPhases::usage =
    "draw2DPhases[phases] = 2d colorwheel or grayscale drawing of given \
    phases in [-pi, pi].";
draw2DLogmags::usage =
    "draw2DLogmags[mags] = 2d log-grayscale drawing of given mags.";
draw2DResids::usage =
    "draw2DResids[resids] = plot 2d resids in two colors.";


Begin["`Private`"];
drawpath[p_] := Module[{pts, i, tmppt, dir},
  pts = {{0,0,0}};
  For[i = 0, i < Length[p], i++;
    dir = p[[i]];
    tmppt = Last[pts];
    tmppt[[Abs[dir]]] = If[dir> 0, tmppt[[Abs[dir]]]+1,
      tmppt[[Abs[dir]]]-1];
    AppendTo[pts, tmppt]];
  Graphics3D[{PointSize[Large], Thick, Point[pts], Line[pts]}, Boxed->False]
  ];

Options[draw2DPhases] = {"Wrapped" -> True};
draw2DPhases[ps_, OptionsPattern[]] := Module[{wrapped, min, max, scale},
  wrapped = OptionValue["Wrapped"];
  If[wrapped,
    ListContourPlot[
      Transpose@ArrayPad[(ps + Pi)/(2 Pi), {{0,1}, {0,1}}],
      ColorFunction -> Function[{z}, Hue[z]],
      ColorFunctionScaling -> False, InterpolationOrder -> 0,
      DataRange->{{0, Dimensions[ps][[1]]}, {0, Dimensions[ps][[2]]}},
      PlotRange->{{-.1, Dimensions[ps][[1]]+.1}, {-.1, Dimensions[ps][[2]]+.1}, All},
      FrameLabel->Map[title, {"x", "t"}], RotateLabel->False,
      frameOpts[], PlotLegends->Automatic],
    ListContourPlot[
      Transpose@ArrayPad[ps, {{0,1}, {0,1}}],
      ColorFunction -> GrayLevel, InterpolationOrder -> 0,
      DataRange->{{0, Dimensions[ps][[1]]}, {0, Dimensions[ps][[2]]}},
      PlotRange->{{-.1, Dimensions[ps][[1]]+.1}, {-.1, Dimensions[ps][[2]]+.1}, All},
      FrameLabel->Map[title, {"x", "t"}], RotateLabel->False,
      frameOpts[], PlotLegends->Automatic]
    ]
  ];

draw2DResids[rs_] := Module[{Lx, Lt, indrs, poscoords, negcoords},
  {Lx, Lt} = Dimensions@rs;
  indrs = Flatten[Table[{i, j, rs[[i, j]]}, {i, Lx}, {j, Lt}], 1];
  poscoords = Cases[indrs, {_, _, 1}][[All, ;; 2]];
  negcoords = Cases[indrs, {_, _, -1}][[All, ;; 2]];
  ListPlot[{poscoords, negcoords}, AxesLabel->Map[title, {"x", "t"}], PlotRange -> All]
  ];

draw2DLogmags[mags_] := Module[{lms},
  lms = Log[mags];
  draw2DPhases[Transpose@lms, "Wrapped" -> False]
  ];

End[]; (* `Private` *)

(* Log and protect everything exported. *)
Print["Loaded " <> Context[] <> " with symbols: ", Names["`*"]];
Protect["`*"];

EndPackage[];