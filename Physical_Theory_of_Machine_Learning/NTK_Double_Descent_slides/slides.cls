\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{slides}[NTUEE SAAD Beamer Class]

% Load the base class
\LoadClass[aspectratio=43]{beamer}
\usefonttheme{professionalfonts}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{parskip}
\RequirePackage{amsmath}
\usepackage{tcolorbox}
\usepackage{tikz}
\usetikzlibrary{backgrounds}
\usetikzlibrary{arrows,shapes}
\usetikzlibrary{tikzmark}
\RequirePackage{amsthm}
\RequirePackage[dvipsnames]{xcolor}
\RequirePackage{CJKutf8}
\newenvironment{chinese}{\begin{CJK}{UTF8}{bkai}}{\end{CJK}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\DeclareOption{maintheme}{
    \definecolor{themecolor}{RGB}{79,1,1}     % NTU color
}

\DeclareOption{bluetheme}{
    \definecolor{themecolor}{RGB}{48,112,179}     % Blue
}

\DeclareOption{redtheme}{
	\definecolor{themecolor}{RGB}{150,0,0}     % red
	}

\DeclareOption{blacktheme}{
	\definecolor{themecolor}{RGB}{0,0,0}     % black
	}

\ExecuteOptions{maintheme}
\ProcessOptions\relax

\definecolor{themecolor1}{RGB}{243,242,243} % gray
\definecolor{themecolor2}{RGB}{234,235,237} % separation lines
% \definecolor{highlighttheme}{RGB}{176,65,62} % red theme
\definecolor{highlighttheme}{RGB}{200,65,62}

%%%%%%%%%%%%%%%%%%

\setbeamertemplate{blocks}[rounded][shadow=true]
\setbeamercolor{block title}{fg=white, bg=blue!60}
\setbeamercolor{block body}{bg=blue!10}
\setbeamercolor{block title alerted}{fg=white, bg=red!70}
\setbeamercolor{block body alerted}{bg=red!10}
\setbeamercolor{block title example}{fg=black, bg=teal!50}
\setbeamercolor{block body example}{bg=teal!10}

% \addtobeamertemplate{footnote}{}{\vspace{2.5ex}}

\setbeamerfont{frametitle}{size=\large \vspace{0.3ex}}

\setbeamercolor{section in toc}{fg=black}
\setbeamercolor{caption name}{fg=black}
\setbeamercolor{title}{fg=white}%
\setbeamercolor{frametitle}{fg=white}%
\setbeamercolor{item projected}{bg=themecolor}%
\setbeamercolor{item}{fg=themecolor}

\setbeamertemplate{items}[bullet]%
\setbeamertemplate{item}[bullet]%
\setbeamertemplate{sections/subsections in toc}[square]%
%%%%%%%%%%%%%%%%%%

\defbeamertemplate*{navigation symbols}{}
% \addtobeamertemplate{navigation symbols}{}{%
%     \usebeamerfont{footline}%
%     \usebeamercolor[fg]{footline}%
%     \hspace{1em}%
%     \insertframenumber/\inserttotalframenumber
% }

%%%%%%%%%%%%%%%%%%


%--------------------
%模版
%--------------------
%標題頁
\defbeamertemplate*{title page}{}[1][left]{%
\strut\vfill\strut%
\vfill\vfill\vfill\vfill%
{\usebeamercolor[fg]{title}\usebeamerfont{title}\inserttitle}%
\vfill%
\begingroup{\ifx\insertsubtitle\empty%
\else%
\usebeamercolor[fg]{title}\usebeamerfont{title}\large{\insertsubtitle}
\vfill%
\fi}\endgroup%
\begin{beamercolorbox}[center,#1]{author}%
\usebeamercolor[bg]{frametitle}\usebeamerfont{author}\insertauthor%
\end{beamercolorbox}\vfill%
\begin{beamercolorbox}[center,#1]{institute}%
\usebeamercolor[fg]{title}\usebeamerfont{institute}\insertinstitute%
\end{beamercolorbox}\vfill%
\begin{beamercolorbox}[center,#1]{date}%
\usebeamercolor[fg]{title}\usebeamerfont{date}\insertdate%
\end{beamercolorbox}%
\vfill\vfill\vfill\vfill\vfill%
}%
%頁腳
\defbeamertemplate*{footline}{footlineoff}{}
\defbeamertemplate*{footline}{footlineon}{%
\leavevmode%
  \ifnum\c@framenumber=1
  \else
  \vspace{-0.004\paperheight}\hspace{-0.01\paperwidth}
	\begin{beamercolorbox}[wd=.56\paperwidth,ht=.025\paperheight,left,dp=2ex]{section in footline}
	\insertsectionnavigationhorizontal{.5\textwidth}{}{}
	\end{beamercolorbox}
	\begin{beamercolorbox}[wd=.25\paperwidth,right,dp=2ex]{shorttitle in footline}
	\quad\insertshorttitle
	\end{beamercolorbox}
	\begin{beamercolorbox}[wd=.1\paperwidth,right,dp=2ex]{shortauthor in footline}
	\insertshortauthor
  \end{beamercolorbox}
	\begin{beamercolorbox}[wd=.08\paperwidth,right,dp=2ex]{page in footline}
	{\insertframenumber{}}/{\inserttotalframenumber\quad}
	\end{beamercolorbox}
  \fi%
}
\setbeamercolor{shorttitle in footline}{fg=white}
\setbeamercolor{section in footline}{fg=white}
\setbeamercolor{shortauthor in footline}{fg=white}
\setbeamercolor{page in footline}{fg=white}

\setbeamerfont{section in head/foot}{size={\fontsize{6}{6}}}
\setbeamercolor{section in head/foot}{fg=white,bg=themecolor}
\setbeamertemplate{section in head/foot shaded}[default][35]%被暗淡的其他標题；調節參數为fg!35!bg

%大綱
\setbeamerfont{section in toc}{size={\fontsize{14}{14}}}
\setbeamercolor{section in toc}{fg=themecolor,bg=white}
\setbeamertemplate{section in toc shaded}[default][35]%被暗淡的其他標题；調節參數为fg!35!bg
%\setbeamerfont{section in toc}{size={\fontsize{6}{6}}}
\setbeamercolor{subsection in toc}{fg=themecolor,bg=white}
\setbeamertemplate{subsection in toc shaded}[default][35]%被暗淡的其他標題；調節參數为fg!35!bg

%beamerbaseauxtemplates.sty
\defbeamertemplate*{subsection in toc}{subsectionhorizontal}%
{\leavevmode%
	\ifnum\inserttocsubsectionnumber=1
	\hspace{3.7ex}%
 	\else
	\fi
    \usebeamercolor[bg]{subsection number projected}%
    \vrule width1ex height1ex\kern1ex\inserttocsubsection}
\setbeamertemplate{subsection in toc}[subsectionhorizontal]


%%%%%%%%%%%%%%%%%%%%
\makeatletter
\@ifclasswith{ctexbeamer}{aspectratio=169}{
		\def\backxscale{4/3}\def\backyscale{1}\def\tumshift{30}
	}{
		\def\backxscale{1.068}\def\backyscale{1.068}\def\tumshift{0}
	}
\@ifclasswith{beamer}{aspectratio=169}{
		\def\backxscale{4/3}\def\backyscale{1}\def\tumshift{30}
	}{
		\def\backxscale{1.068}\def\backyscale{1.068}\def\tumshift{0}
	}
\makeatother


%%%%%%%%%%%%%%%%%%%

\newcommand{\maketitleframe}{
	%取消footline,sidebar
	\setbeamertemplate{footline}[footlineoff]
	%%設置本命令以后的背景如下
	\usebackgroundtemplate{%
			\input{./theme/cover.tex}
	}%
	%封面頁
	\begin{frame}%
		\maketitle%
	\end{frame}%
	%設置本命令以後的背景如下
	\usebackgroundtemplate{%
			\input{./theme/background.tex}
	}%
}

%%%%%%%%%%%%%%%%%%%

\AtBeginSection[]
{
  \begin{frame}
    \frametitle{Table of Contents}
    \tableofcontents[currentsection,hideothersubsections]
  \end{frame}
}
\AtBeginSubsection[]
{
  \begin{frame}
    \frametitle{Table of Contents}
    \tableofcontents[currentsection,currentsubsection,subsectionstyle=show/shaded/hide]
  \end{frame}
}
