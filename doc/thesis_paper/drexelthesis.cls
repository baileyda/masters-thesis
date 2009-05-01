%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Drexel University Thesis LaTeX2e Class defintion.
% Modified by Jeff Abrahamson    May 2005
% Patched by C.Y. Ip             2005-02-12
% Patched by Christopher S. Dahn 2004-04-19
% Patched by Christopher D. Cera 2003-03-10
% Patched by Mitchell Peabody in 2002
% Patched by David T. McWherter  2000-07-13
% Written by Vincent A. Cicirello 1999-02-13 based on:
% University of Toronto Thesis LaTex2e Class definition.
% Written by Robert Bernecky 1997-07-10, based on
% the thesis.sty written at UofT by fpitt.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{drexelthesis}[2000/07/13 Drexel University Thesis class]

\RequirePackage{setspace}

% The following options are what you get if you do not specify ANY options.
%% E.g., \documentclass{drexelthesis}
%% Basically, it is your camera copy.

\ExecuteOptions{letterpaper,oneside,12pt,onecolumn,final,openany}

%% If you want a cheap-and-cheerful draft copy, try options such
%% \documentclass[draft,twoside,11pt]{drexelthesis} 

%% Set up handling of "draft" option. 

\DeclareOption{draft}{%
  \PassOptionsToPackage{all}{draftcopy}% 
  \PassOptionsToClass{draft}{report}% 
  \AtEndOfClass{%
    \RequirePackage{draftcopy}%
    %% I wanted to print timestamp of each file, but I don't see
    %% how to do that...
    \draftcopyName{DRAFT: \today}{60} 
    % text for overprinting if in "draft" mode
  }}

% Pass ALL options to report.cls.
\DeclareOption*{%
  \PassOptionsToClass{\CurrentOption}{report}
}

\ProcessOptions

\LoadClass{report} 

%% Load packages...
\usepackage{indentfirst}

%% Use Geometry package to set margins precisely
%%
%% Dmitriy / Horace used this definition:
%\usepackage[includehead, top=0.75in,left=1.5in,right=1in,bottom=1in,head=1em]{geometry}
%% Top margin includes head=0.75in, ie page number starts at 0.75in 
%%
%% Debian's TeX installation apparently has different page boundaries,
%% so I fiddle with this a bit.
\usepackage[includehead, head=1em, top=0.75in, left=1.5in, right=1in, bottom=1in]{geometry}
%% For some reason, I need to shift down by a half inch.
%% --Jeff A
%\addtolength{\topmargin}{2.5in}


%% Manually adjust the header, Goal: pagenumber at 0.75in below the top,
%% headheight+headsep=0.25in
%\setlength{\headheight}{0.75in} %% need 1 line of header for page number

\usepackage[tableposition=top,aboveskip=1em, belowskip=0em]{caption}
\usepackage[TABTOPCAP]{subfigure}
\usepackage[subfigure]{tocloft}
\usepackage{url}

% Set up environments for thesis sections.
% New or renew Commands

%% If necessary, lower page numbers a bit on the page
%\newcommand

% clearemptydoublepage from page 92 of The LaTeX Companion 1993. 
\newcommand{\clearemptydoublepage}{\newpage{\pagestyle{empty}\cleardoublepage}}

%%% ucip: produce an empty page after prelim and set page counter to 1
\newcommand\blankpage{\clearpage\vspace*{1em}\thispagestyle{empty}\newpage}

% Headings formating ... chapter section etc ...

\renewcommand{\@chapapp}{}

\renewcommand{\chapter}{\if@openright\cleardoublepage\else\clearpage\fi
  \thispagestyle{plain}%
  \global\@topnum\z@
  \@afterindentfalse
  \secdef\@chapter\@schapter}


\renewcommand{\section}{%
  \@startsection {section}{1}{\z@}%
  {-3.5ex \@plus -1ex \@minus -.2ex}%
  {2.3ex \@plus.2ex}%
  {\normalfont\normalsize\bfseries}}

\renewcommand{\subsection}{%
  \@startsection{subsection}{2}{\z@}%
  {-3.25ex\@plus -1ex \@minus -.2ex}%
  {1.5ex \@plus .2ex}%
  {\normalfont\bfseries}}

\renewcommand{\subsubsection}{%
  \@startsection{subsubsection}{3}{\z@}%
  {-3.25ex\@plus -1ex \@minus -.2ex}%
  {1.5ex \@plus .2ex}%
  {\normalfont\normalsize\bfseries}}

\renewcommand{\paragraph}{%
  \@startsection{paragraph}{4}{\z@}%
  {3.25ex \@plus1ex \@minus.2ex}%
  {-1em}%
  {\normalfont\normalsize\bfseries}}

\renewcommand{\subparagraph}{%
  \@startsection{subparagraph}{5}{\parindent}%
  {3.25ex \@plus1ex \@minus .2ex}%
  {-1em}%
  {\normalfont\normalsize\bfseries}}

\renewcommand{\chapter}{%
  \if@openright\cleardoublepage\else\clearpage\fi
  \thispagestyle{plain}%
  \global\@topnum\z@
  \@afterindentfalse
  \secdef\@chapter\@schapter}

\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
  \refstepcounter{chapter}%
  \typeout{\@chapapp\space\thechapter.}%
  \phantomsection
  \addcontentsline{toc}{chapter}%
  {\protect\numberline{\thechapter}#1}%
  \else
  \phantomsection
  \addcontentsline{toc}{chapter}{#1}%
  \fi
  \chaptermark{#1}%
  \if@twocolumn
  \@topnewpage[\@makechapterhead{#2}]%
  \else
  \@makechapterhead{#2}%
  \@afterheading
  \fi}

\def\@makechapterhead#1{%
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
    \centering \bfseries \@chapapp\space \thechapter.\ 
    \fi
    \interlinepenalty\@M
    \centering \bfseries #1\par\nobreak
    \vskip 40\p@
  }}

\def\@schapter#1{\if@twocolumn
  \@topnewpage[\@makeschapterhead{#1}]%
  \else
  \@makeschapterhead{#1}%
  \@afterheading
  \fi}

\def\@makeschapterhead#1{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright
    \normalfont
    \interlinepenalty\@M
    \centering \bfseries  #1\par\nobreak
    \vskip 40\p@
  }}

%%% TOC LOF LOT Formatting...
\renewcommand{\cftdotsep}{1}
\renewcommand{\cfttoctitlefont}{\hfill\normalfont\bfseries}
\renewcommand{\cftaftertoctitle}{\hfill}
\renewcommand{\cftchapaftersnum}{.}
\renewcommand{\cftchapleader}{\cftsecleader}
\renewcommand{\cftchapfont}{\normalfont}
\setlength{\cftbeforechapskip}{\cftbeforesecskip}

%%% Skip one line after each entry to make it double space.
%% \setlength{\cftbeforechapskip}{0.667em}
%% \setlength{\cftbeforesecskip}{0.667em}
%% \setlength{\cftbeforesubsecskip}{0.667em}
%% \setlength{\cftbeforefigskip}{0.667em}
%% \setlength{\cftbeforetabskip}{0.667em}

\def\ssp{\def\baselinestretch{1.0}\large\normalsize}
\renewcommand{\cftfigfont}{\setstretch {1}\normalsize}
\renewcommand{\cftfigafterpnum}{\par}


\setlength{\cftfigindent}{0pt}
\setlength{\cfttabindent}{0pt}

\renewcommand{\cftchappagefont}{}


%%%%%% ucip: replaced all \drexelborder with \drexelmainborder for tables
\newcommand{\mytableofcontents}{%
  \begin{doublespace}%singlespace}
  \if@twocolumn
  \@restonecoltrue\onecolumn
  \else
  \@restonecolfalse
  \fi
  {\centering \section*{Table of Contents}}%
  \@starttoc{toc}%
  \if@restonecol\twocolumn\fi
  \newpage
  \end{doublespace}%singlespace}
	\clearpage
}


\newcommand{\mylistoffigures}{
  \clearemptydoublepage
  \phantomsection
  \addcontentsline{toc}{chapter}{List of Figures}
  \begin{singlespace}
		{
	  \if@twocolumn
  		\@restonecoltrue\onecolumn
  	\else
  		\@restonecolfalse
  	\fi
		{\centering \section*{List of Figures}}%
  	\setlength{\parskip}{\baselineskip}
  	\@starttoc{lof}%
  	\if@restonecol\twocolumn\fi
  	\newpage
  	}
  \end{singlespace}
	\clearpage
}

\newcommand{\mylistoftables}{
  \phantomsection
  \addcontentsline{toc}{chapter}{List of Tables}
  \begin{singlespace}
	  {
	  \if@twocolumn
  		\@restonecoltrue\onecolumn
  	\else
  		\@restonecolfalse
 		\fi
  	{\centering \section*{List of Tables}}
  	\setlength{\parskip}{\baselineskip}
  	\@starttoc{lot}%
  	\if@restonecol\twocolumn\fi
  	\newpage
  	}
  \end{singlespace}
	\clearpage
}


\newcommand{\degree}[1]{\ifx\empty#1\else\gdef\@degree{#1}\fi}
\newcommand{\degreearea}[1]{\ifx\empty#1\else\gdef\@degreearea{#1}\fi}
\renewcommand{\author}[1]{\ifx\empty#1\else\gdef\@author{#1}\fi}
\newcommand{\advisor}[1]{\ifx\empty#1\else\gdef\@advisor{#1}\fi}
\renewcommand{\title}[1]{\ifx\empty#1\else\gdef\@title{#1}\fi}
\newcommand{\gyear}[1]{\ifx\empty#1\else\gdef\@gyear{#1}\fi}
\newcommand{\gmonth}[1]{\ifx\empty#1\else\gdef\@gmonth{#1}\fi}

\global\def\@degree{\mbox{}}
\global\def\@degreearea{\mbox{}}
\global\def\@author{\mbox{}}
\global\def\@advisor{\mbox{}}
\global\def\@title{\mbox{}}
\global\def\@gyear{\number\year} % Default year is current year
\global\def\@gmonth{\mbox{}}

% redefine the plain pagestyle so that page numbers are in the upper
% right hand corner 
% this is to deal with commands such as \tableofcontents that like to
% switch the pagestyle to plain

\renewcommand{\ps@plain}{%
  \renewcommand{\@mkboth}{\@gobbletwo}%
  \renewcommand{\@evenhead}{\reset@font\rmfamily\hfill\thepage}%
  \renewcommand{\@oddhead}{\reset@font\rmfamily\hfill\thepage}%
  \renewcommand{\@evenfoot}{}%
  \renewcommand{\@oddfoot}{}}

\renewcommand{\maketitle}{
  \begin{titlepage}
    \vspace*{\fill} %%use 2 vertical fill to center the title
    \noindent    
    \begin{minipage}{5.5in} %% 5.5in make right margin=1.5in for centering
      \begin{doublespace}
        \centering
      	  \begin{singlespace}
      			\textbf{{\@title}}\\
      		\end{singlespace}
        \vspace*{1em}
        A Thesis \\
        Submitted to the Faculty \\
        of \\
        Drexel University \\
        by \\
        {\@author}\\
        in partial fulfillment of the \\
				requirements for the degree \\
				of \\
				{\@degree} in {\@degreearea} \\
        {\@gmonth} {\@gyear} \\
      \end{doublespace}
    \end{minipage}
    \vspace*{\fill}
  \end{titlepage}
  \clearpage
}%\maketitle

\newcommand{\copyrightpage}{
  \clearpage
	\setcounter{page}{1}
  \thispagestyle{empty}
  \vspace*{\fill}
  \noindent
  \begin{minipage}{5.5in}
    \begin{singlespace}
      \centering
      \copyright~Copyright \@gyear \\
      \@author.  All Rights Reserved.
    \end{singlespace}
  \end{minipage}
  \vspace*{\fill}
%  \addtocounter{page}{1}
  \clearpage
}

\newcommand{\nicecopyrightpage}{
  \clearpage
	\setcounter{page}{1}
  \thispagestyle{empty}
  \vspace*{\fill}
  \noindent
  \begin{minipage}{5.5in}
    \begin{singlespace}
      \centering
      \copyright~Copyright \@gyear \\
      \@author.
      \begin{quote}
        This work is licensed under the terms of the Creative
        Commons Attribution-ShareAlike license.  The license is
        available at \url{http://creativecommons.org/licenses/by-sa/2.0/}.
      \end{quote} 
    \end{singlespace}
  \end{minipage}
  \vspace*{\fill}
  \addtocounter{page}{0}
  \clearpage
}


%%% End of New Commands...

%%% New or renew environments

%%% Pad lines of space in figure and tables 3 on top 1 on top (2 other
%%% comes from the caption skip)

\renewenvironment{thebibliography}[1]{
  \clearpage
  \begin{center}    
    \phantomsection
    \addcontentsline{toc}{chapter}{Bibliography}
 		\section*{\bibname}
  \end{center}
  \vspace{1em}
  \singlespacing
  \list{\@biblabel{\@arabic\c@enumiv}} %
  {\settowidth\labelwidth{\@biblabel{#1}}%
    \leftmargin\labelwidth
    \advance\leftmargin\labelsep
    \@openbib@code
    \usecounter{enumiv}%
    \let\p@enumiv\@empty
    \renewcommand\theenumiv{\@arabic\c@enumiv}}%
  \sloppy
  \singlespacing
  \clubpenalty4000
  \@clubpenalty \clubpenalty
  \widowpenalty4000%
  \sfcode`\.\@m}
{\def\@noitemerr
  {\@latex@warning{Empty `thebibliography' environment}}%
  \endlist \clearpage}

\renewenvironment{abstract}{
  \phantomsection
  \addcontentsline{toc}{chapter}{Abstract} 
  \begin{singlespace}
    {\centering
      \section*{Abstract}
      \vspace*{-1em}
      {\@title}\\[3mm]
      {\@author}\\
      {\smallskip Advisor: \@advisor}, Ph.D.\\
      \vspace*{4em}
    }
  \end{singlespace}
  \begingroup
}{\clearpage\endgroup}

\newenvironment{acknowledgments}{
  \clearemptydoublepage
  \begin{center}
    \section*{Acknowledgments}
  \end{center}
  \begingroup
}{\clearpage\endgroup}

\newenvironment{dedications}{
  \clearemptydoublepage 
  \begin{center}
    \section*{Dedications}
  \end{center}
  \begingroup
}{\clearpage\endgroup}

\newenvironment{gratuitousquote}{
  \clearemptydoublepage 
  \flushright
  \vbox to 1.8in{}
  \begingroup
}{\clearpage\endgroup}

\newenvironment{preliminary}{\pagenumbering{roman}
  \addtocounter{page}{1}\begingroup}{\blankpage\endgroup}

\newenvironment{thesis}{\pagenumbering{arabic}
  \setcounter{page}{1}\begingroup}{\blankpage\endgroup}

%%% End of New or renew environments


%% Footnotes are always single-spaced.
%% \let\@@footnotetext=\@footnotetext
%% \def\@footnotetext{\begin{spacing}{1}\@@footnotetext\end{spacing}}


%%% Misc page style and exec default options

%% Define the "thesis" page style.
\if@twoside % If two-sided printing.
\def\ps@thesis{\let\@mkboth\markboth
  \def\@oddfoot{}
  \let\@evenfoot\@oddfoot
  \def\@oddhead{\reset@font\rmfamily\hfil\thepage}
  \def\@evenhead{\reset@font\rmfamily\hfil\thepage}
  
  \def\chaptermark##1{\markright{\ifnum \c@secnumdepth >\m@ne
      Chapter\ \thechapter. \ \fi ##1}}}
\else % If one-sided printing.
\def\ps@thesis{\let\@mkboth\markboth
  \def\@oddfoot{}
  \def\@oddhead{\reset@font\rmfamily\hfil\thepage}
  
  \def\chaptermark##1{\markright{\ifnum \c@secnumdepth >\m@ne
      Chapter\ \thechapter. \ \fi ##1}}}
\fi


% Set up page layout.
\pagestyle{plain}

% Marginal notes
% \setlength{\marginparwidth}{.75in} % width of marginal notes
% \setlength{\marginparsep}{.125in} % space between marginal notes and text
%% \setlength{\headheight}{0cm} % space between marginal notes and text
%% \setlength{\headsep}{0in} % space between marginal notes and text
%% \addtolength{\headsep}{-1em} % space between marginal notes and text

% Make each page fill up the entire page. comment this out if you
% prefer. 
% \flushbottom


%%% Add 3 lines of space for all the float 
\setlength{\intextsep}{3.5em}
\setlength{\floatsep}{3.5em}
\setlength{\textfloatsep}{3.5em}
\setlength{\dblfloatsep}{3.5em}
\setlength{\dbltextfloatsep}{3.5em}
