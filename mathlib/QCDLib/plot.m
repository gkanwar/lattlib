(* ::Package:: *)
(* Plotting functionality *)
BeginPackage["QCDLib`plot`", {"QCDLib`util`"}];

(* Unprotect and clear everything. *)
Unprotect["`*"];
ClearAll["`*"];

(* Define everything implicitly by giving usages. *)
bigErrBarFn::usage =
    "bigErrBarFn[halfWidth] = function to plot error bar with end width 2d.";
constFitEpilog::usage =
    "constFitEpilog[fitMean, fitErr, window] = list of epilog actions to plot constant line fit.";
frameOpts::usage =
    "frameOpts[] = list of standard plot sizing and frame options.";
title::usage =
    "title[str] = str in plot title format.";
errorListLogPlot::usage =
    "errorListLogPlot[series] = log plot with error bars. Series should either be\
    a single list of {mean,err} tuples, or a list of series, each of which are a list of\
    {mean,err} tuples.";


Begin["`Private`"];

Options[bigErrBarFn] = {"Directives" -> {}};
bigErrBarFn[d_, OptionsPattern[]] := Function[{coords, errs},
   Module[{out},
          out = Join[OptionValue["Directives"], {
           Line[{coords + {0, errs[[2, 1]]}, 
                 coords + {0, errs[[2, 2]]}}]}];
          If[d > 0,
             out = Join[out, {
                 Line[{coords + {-d, errs[[2, 1]]}, coords + {d, errs[[2, 1]]}}],
                 Line[{coords + {-d, errs[[2, 2]]}, coords + {d, errs[[2, 2]]}}]}]];
     out]];

constFitEpilog[fitMean_, fitErr_, {ti_,tf_}] := {
  Pink, Opacity[0.5], Rectangle[{ti, fitMean-fitErr}, {tf, fitMean+fitErr}],
  Red, Thick, Opacity[1.0], Line[{{ti, fitMean}, {tf, fitMean}}]
  };

frameOpts[] = {
  ImageSize -> 500, Frame -> True,
  FrameStyle -> Thick, FrameTicksStyle -> Directive[18]
  };

title[str_] := Style[str, Bold, Black, 18];

Options[errorListLogPlot] = {
  "LogPad" -> 0.05, "LogHardPad" -> 0.10,
  "YRange" -> Automatic,
  "Colors" -> mmaColors, "MeanJoined" -> False,
  "ErrorMarker" -> {Graphics[{Thick, Line[{{-0.5,0},{0.5,0}}]}], 0.03},
  "FillingStyle" -> {},
  "MainMarker" -> Automatic};
errorListLogPlot[series_, opts:OptionsPattern[{errorListLogPlot, ListLogPlot}]] :=
    Module[{allseries, logplotmax, logplotmin, loghardmax, loghardmin, logrange,
      logpad, loghardpad, xpts, mpts, epts, pts, xmax, xmin, xspan, yrange, colors, extraPlotOpts},
      colors = OptionValue["Colors"];
      (* allseries has dims {nseries, npts, 2} or {nseries, npts, 3} *)
      allseries = If[Length@Dimensions@series == 2, {series}, series];
      allseries = Table[
        Table[
          If[Length[allseries[[i,j]]] == 3, allseries[[i,j]],
            Join[{j}, allseries[[i,j]]]],
          {j, Length[allseries[[i]]]}],
        {i, Length@allseries}];
      {xpts, mpts, epts} = Table[
        Table[allseries[[i,j,piece]], {j, Length[allseries[[i]]]}],
        {piece, 3}, {i, Length@allseries}];
      (* {xpts, mpts, epts} = Transpose[allseries, {2,3,1}]; *)
      logpad = OptionValue["LogPad"];
      loghardpad = OptionValue["LogHardPad"];
      logplotmax = Max@Abs@mpts;
      logplotmin = Min@Abs@mpts;
      logrange = Log[logplotmax] - Log[logplotmin];
      logplotmax *= Exp[logpad*logrange];
      logplotmin /= Exp[logpad*logrange];
      yrange = If[MatchQ[OptionValue["YRange"], Automatic],
        {logplotmin, logplotmax}, OptionValue["YRange"]];
      loghardmax = logplotmax * Exp[loghardpad * logrange];
      loghardmin = logplotmin / Exp[loghardpad * logrange];
      {xmin, xmax} = {Min[xpts], Max[xpts]};
      xspan = xmax-xmin;
      {xmin, xmax} = {xmin-xspan/10, xmax+xspan/10};
      ptsMean = Table[{mpts[[i]]}, {i, Length@mpts}];
      ptsErr = Table[{mpts[[i]] - epts[[i]],
        mpts[[i]] + epts[[i]]}, {i, Length@mpts}];
      ptsMean = Map[
        If[# > loghardmax, loghardmax, 
          If[# < loghardmin, loghardmin, #]] &, ptsMean, {3}];
      ptsErr = Map[
        If[# > loghardmax, loghardmax, 
          If[# < loghardmin, loghardmin, #]] &, ptsErr, {3}];
      colorsMean = colors[[;;Length@ptsMean]];
      colorsErr = Join@@Map[ConstantArray[#, 2]&, colorsMean];
      markersMean = OptionValue["MainMarker"];
      If[MatchQ[markersMean, Automatic],
        markersMean = mmaPlotMarkers[[;;Length@ptsMean]]];
      markersErr = ConstantArray[OptionValue["ErrorMarker"], 2 Length@ptsErr];
      fillingsErr = Table[2 i -> {2 i - 1}, {i, Length@ptsErr}];
      extraPlotOpts = getOptsFor[ListLogPlot, opts];
      extraPlotOptsNoLegend = FilterRules[extraPlotOpts, Except[PlotLegends]];
      If[OptionValue["MeanJoined"],
        AppendTo[extraPlotOpts, Joined->True]];
      tracesErr = Join@@ptsErr;
      tracesMean = Join@@ptsMean;
      tracesMean = Table[
        Transpose@{xpts[[i]], tracesMean[[i]]}, {i, Length@tracesMean}];
      tracesErr = Table[
        Transpose@{xpts[[Floor[(i+1)/2]]], tracesErr[[i]]}, {i, Length@tracesErr}];
      Show[{
        ListLogPlot[tracesErr,
          PlotRange -> {{xmin, xmax}, yrange},
          Filling -> fillingsErr, FillingStyle ->
          Directive@Join[{Opacity[1.0]}, OptionValue["FillingStyle"]],
          PlotStyle -> colorsErr, PlotMarkers -> markersErr,
          extraPlotOptsNoLegend],
        ListLogPlot[tracesMean,
          PlotRange -> {{xmin, xmax}, yrange},
          PlotStyle -> colorsMean, PlotMarkers -> markersMean,
          extraPlotOpts
          ]
        }]
      ];

End[]; (* `Private` *)


(* Log and protect everything exported. *)
Print["Loaded " <> Context[] <> " with symbols: ", Names["`*"]];
Protect["`*"];

EndPackage[];