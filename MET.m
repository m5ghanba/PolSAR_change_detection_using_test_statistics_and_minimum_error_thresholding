function varargout = MET(varargin)
% MET MATLAB code for MET.fig
%      MET, by itself, creates a new MET or raises the existing
%      singleton*.
%
%      H = MET returns the handle to a new MET or the handle to
%      the existing singleton*.
%
%      MET('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MET.M with the given input arguments.
%
%      MET('Property','Value',...) creates a new MET or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before MET_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to MET_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help MET

% Last Modified by GUIDE v2.5 08-Oct-2014 12:15:46

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @MET_OpeningFcn, ...
                   'gui_OutputFcn',  @MET_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before MET is made visible.
function MET_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to MET (see VARARGIN)

% Choose default command line output for MET
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes MET wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = MET_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global t_s d ts L r r_m c_m T_M TS TSS val_type n1m1 n1w1 n1w2 n1m2 n1w0 n2m1 n2w1 n2w2 n2m2 n2w0 n1nu2 n1nu3 n2nu2 n2nu3
%save('T_MM.mat','T_M')
tic
format long
val_method(1,1)=get(handles.radiobutton46,'value');
val_method(2,1)=get(handles.radiobutton47,'value') ;
if val_method(1)==1
%minimizng criterion function

%histogram of quantized trace image
%ts=TSS;
maxts=max(max(ts));
%max(max(ts));
h1=zeros(maxts+1,1);
h2=zeros(maxts+1,1);
h3=zeros(maxts+1,1);
kk=zeros(maxts+1,1);

for k=0:maxts
    kk(k+1)=k;
    for l=1:r_m
        for m=1:c_m
            if k==0
                kk(k+1)=.1;
                if (ts(l,m))==.1 && T_M(l,m)~=0
                    h1(k+1)=h1(k+1)+1;
                    if T_M(l,m)==127.5
                        h2(k+1)=h2(k+1)+1;
                    end
                    if T_M(l,m)==255
                        h3(k+1)=h3(k+1)+1;
                    end
                end
            end
            if (ts(l,m))==k && T_M(l,m)~=0
                h1(k+1)=h1(k+1)+1;
                if T_M(l,m)==127.5
                    h2(k+1)=h2(k+1)+1;
                end
                if T_M(l,m)==255
                    h3(k+1)=h3(k+1)+1;
                end                
            end
        end
    end
end

%h=h1/(r_m*c_m);
h=h1/(size(find(T_M~=0),1));
h_test_nc=h2/(size(find(T_M==127.5),1));%manateghe test marboot be no change
h_test_c=h3/(size(find(T_M==255),1));%manateghe test marboot be no change
axes(handles.axes5);
 k_pr=10*log10(kk);
% h_pr=10*log10(h);
bar(kk,h1,'b');
 
 figure(1);
 clf
 linenu = 1.5;fs=20;
%[fA,xA] = [h,kk];
% [fB,xB] = ksdensity(Mj2,'npoints',10000);
% [fC,xC] = ksdensity(Mj5,'npoints',10000);
log_h1=10*log10(h1);
 xvector=kk';
 fvector=h1';
 plot(xvector, fvector, 'LineWidth',linenu);
 %legend({'Histogram of TS image'},'FontSize',fs)
 xlabel(['Gray level'], 'fontsize',fs)
 ylabel('Frequency', 'fontsize',fs)
set(gca,'FontSize',fs);
% hold on
% aa=22;
% bb=300;
% plot(aa,bb)

%J:ceriterion function. 
%taw=quntized trace statistic values.
%P_0_taw:prior probability of H0 hypothesis. P_1_taw:prior probability of H1 hypothesis. 
%H0_2ndterm:2nd term of ceriterion function related to H0. H1_2ndterm:2nd term of cerioterion function related to H1. 
%min_J:mininmum of ceriterion function. min_error_taw:desired minimum error threshold
J=1.e10*ones(maxts+1,1);
oerror=1.e10*ones(maxts+1,1);%Overall error vector
Thre=1.e10*ones(maxts+1,1);%Overall error vector vs J function plot vs threshold
taw_vales=zeros(maxts+1,1);
min_J=zeros(maxts+1,1);
ind=1;
%(maxts-(maxts/3))
for taw=0:100%(maxts-(maxts/2))
%     alpha=37;
%     lambda=3;
%     mu=8;
%parameter estimation

value1_1(1,1)=get(handles.radiobutton11,'value');%parameter estimation method
value1_1(2,1)=get(handles.radiobutton12,'value');
value1_1(3,1)=get(handles.radiobutton13,'value');
val_pdf(1,1)=get(handles.radiobutton14,'value');
val_pdf(2,1)=get(handles.radiobutton15,'value');
val_pdf(3,1)=get(handles.radiobutton53,'value');
val_pdf(4,1)=get(handles.radiobutton54,'value');
val_pdf(5,1)=get(handles.radiobutton16,'value');
%moment-based

if value1_1(1)==1
    

if val_pdf(1)==1
    %% PE of fisher pdf by moment method 
    
n1sum_ts=0;%This parameter is equal to summation of (ts). (class n1:"no change" class)
n1sum_ts2=0;%This parameter is equal to summation of(ts)^2.
n1sum_ts3=0;%This parameter is equal to summation of (ts)^3.
n1num_sum=0;
n2sum_ts=0;%This parameter is equal to summation of (ts). (class n2:"change" class)
n2sum_ts2=0;%This parameter is equal to summation of(ts)^2.
n2sum_ts3=0;%This parameter is equal to summation of (ts)^3.
n2num_sum=0;
for i=1:r_m
    for j=1:c_m
        if ts(i,j)<=taw || taw==0
        n1sum_ts=n1sum_ts+(ts(i,j));
        n1sum_ts2=n1sum_ts2+((ts(i,j)))^2;
        n1sum_ts3=n1sum_ts3+((ts(i,j)))^3;
        n1num_sum=n1num_sum+1;
        elseif ts(i,j)>taw
        n2sum_ts=n2sum_ts+(ts(i,j));
        n2sum_ts2=n2sum_ts2+((ts(i,j)))^2;
        n2sum_ts3=n2sum_ts3+((ts(i,j)))^3;
        n2num_sum=n2num_sum+1;
        end
    end
end
n1m1=n1sum_ts/n1num_sum;
n1m2=n1sum_ts2/n1num_sum;
n1m3=n1sum_ts3/n1num_sum;
if ~(taw==0 || taw==255) 
n2m1=n2sum_ts/n2num_sum;
n2m2=n2sum_ts2/n2num_sum;
n2m3=n2sum_ts3/n2num_sum;
end

n1mu=(2*n1m1*((n1m2^2)-n1m1*n1m3))/(4*(n1m2^2)-3*n1m1*n1m3-(n1m1^2)*n1m2);
n1alpha=(2*n1m1*(n1m2^2-n1m1*n1m3))/(2*n1m3*(n1m1^2)-n1m2*n1m3-n1m1*(n1m2^2));
n1lambda=(3*n1m1*n1m3-4*(n1m2^2)+(n1m1^2)*n1m2)/((n1m1^2)*n1m2+n1m1*n1m3-2*(n1m2^2));
if ~(taw==0 || taw==255) 
n2mu=(2*n2m1*((n2m2^2)-n2m1*n2m3))/(4*(n2m2^2)-3*n2m1*n2m3-(n2m1^2)*n2m2);
n2alpha=(2*n2m1*(n2m2^2-n2m1*n2m3))/(2*n2m3*(n2m1^2)-n2m2*n2m3-n2m1*(n2m2^2));
n2lambda=(3*n2m1*n2m3-4*(n2m2^2)+(n2m1^2)*n2m2)/((n2m1^2)*n2m2+n2m1*n2m3-2*(n2m2^2));
end
if taw==0 || taw==255 
n2alpha=n1alpha;
n2lambda=n1lambda;
n2mu=n1mu;
end
if n1lambda<0 || n2lambda<0
    msgbox('There does not exsit all the first three moments of the pdf!','not suitable PE method','error');
    break
end
elseif val_pdf(2)==1
    %% PE of Gamma by moment method
    

n1sum_ts=0;%This parameter is equal to summation of (ts). (class n1:"no change" class)
n1sum_ts2=0;%This parameter is equal to summation of(ts)^2.
n1num_sum=0;
n2sum_ts=0;%This parameter is equal to summation of (ts). (class n2:"change" class)
n2sum_ts2=0;%This parameter is equal to summation of(ts)^2.
n2num_sum=0;
for i=1:r_m
    for j=1:c_m
        if ts(i,j)<=taw || taw==0
        n1sum_ts=n1sum_ts+(ts(i,j));
        n1sum_ts2=n1sum_ts2+(ts(i,j))^2;
        n1num_sum=n1num_sum+1;
        elseif ts(i,j)>taw
        n2sum_ts=n2sum_ts+(ts(i,j));
        n2sum_ts2=n2sum_ts2+((ts(i,j)))^2;
        n2num_sum=n2num_sum+1;
        end
    end
end
n1m1=n1sum_ts/n1num_sum;
n1m2=n1sum_ts2/n1num_sum;
if ~(taw==0 || taw==255) 
n2m1=n2sum_ts/n2num_sum;
n2m2=n2sum_ts2/n2num_sum;
end
n1mu=n1m1;
n1l=1/((n1m2/(n1m1^2))-1);
if ~(taw==0 || taw==255) 
n2mu=n2m1;
n2l=1/((n2m2/(n2m1^2))-1);
end
if taw==0 || taw==255 
n2mu=n1mu;
n2l=n1l;
end    
end
%log-cumulant based

elseif value1_1(2)==1
 
if val_pdf(1)==1
    %% PE of fisher pdf by log cumulant method
    
    
    
% n1=0;n2=0;n3=0;n4=0;n5=0;n6=0;n7=0;n8=0;n9=0;n10=0;    
% for i=1:r_m
%     for j=1:c_m
%         if ts(i,j)>0 && ts(i,j)<1
%             n1=n1+1;
%         end
%         if ts(i,j) >=1 && ts(i,j)<2
%             n2=n2+1;
%         end
%         if ts(i,j)>=2 && ts(i,j)<3
%             n10=n10+1;
%         end
%         if ts(i,j)>=3 && ts(i,j)<4
%             n3=n3+1;
%         end
%         if ts(i,j)>=4 && ts(i,j)<5
%             n4=n4+1;
%         end
%         if ts(i,j)>=5 && ts(i,j)<10
%             n5=n5+1;
%         end
%         if ts(i,j)>=10 && ts(i,j)<50
%             n6=n6+1;
%         end
%         if ts(i,j)>=50 && ts(i,j)<200
%             n7=n7+1;
%         end
%         if ts(i,j)>=200 && ts(i,j)<500
%             n8=n8+1;
%         end 
%         if ts(i,j)>=500 
%             n9=n9+1;
%         end
%     end
% end
% n1
% n2
% n10
% n3
% n4
% n5
% n6
% n7
% n8
% n9
% pause


n1sum_ts=0;
n2sum_ts=0;
n1sum_log_ts=0;%This parameter is equal to summation of log(ts).(class n1) 
n1sum_log_ts2=0;%This parameter is equal to (summation of (log(ts)^2) minus nu1^2.
n1sum_log_ts3=0;%This parameter is equal to summation of (log(ts)-n1nu1)^3.
n1num_sum=0;
n2sum_log_ts=0;%This parameter is equal to summation of log(ts). (class n2)
n2sum_log_ts2=0;%This parameter is equal to summation of (log(ts)-n2nu1)^2.
n2sum_log_ts3=0;%This parameter is equal to summation of (log(ts)-n2nu1^3.
n2num_sum=0;
for i=1:r_m
    for j=1:c_m
        if ts(i,j)<=taw || taw==0
                                    if T_M(i,j)~=0
        n1sum_log_ts=n1sum_log_ts+log(ts(i,j));
        n1sum_ts=n1sum_ts+ts(i,j);
                                    end
        %n1sum_log_ts2=n1sum_log_ts2+(log10(ts(i,j)))^2;
        %n1sum_log_ts3=n1sum_log_ts3+(log10(ts(i,j)))^3;
        n1num_sum=n1num_sum+1;
        elseif ts(i,j)>taw
                                    if T_M(i,j)~=0
        n2sum_log_ts=n2sum_log_ts+log(ts(i,j));
        n2sum_ts=n2sum_ts+ts(i,j);
                                    end
        %n2sum_log_ts2=n2sum_log_ts2+(log10(ts(i,j)))^2;
        %n2sum_log_ts3=n2sum_log_ts3+(log10(ts(i,j)))^3;
        n2num_sum=n2num_sum+1;
        end
    end
end
% n1sum_ts
% n1num_sum
% n1m1
% n2sum_ts
% n2num_sum
if n1num_sum ~=0
n1m1=n1sum_ts/n1num_sum;
n2m1=n1m1;
n1nu11=n1sum_log_ts/n1num_sum;
% n1nu12=n1sum_log_ts2/n1num_sum;
% n1nu13=n1sum_log_ts3/n1num_sum;
n1nu1=n1nu11;
% n1nu2=n1nu12-(n1nu11^2);
% n1nu3=n1nu13-3*n1nu11*n1nu12+2*(n1nu11^3);
n2nu11=n1nu11;
% n2nu12=n2sum_log_ts2/n2num_sum;
% n2nu13=n2sum_log_ts3/n2num_sum;
n2nu1=n2nu11;
% n2nu2=n2nu12-(n2nu11^2);
% n2nu3=n2nu13-3*n2nu11*n2nu12+2*(n2nu11^3);
end
if ~(taw==0 || taw==maxts) 
n2m1=n2sum_ts/n2num_sum;
n2nu11=n2sum_log_ts/n2num_sum;
n2nu1=n2nu11;
end
% n1nu1
% n2m1
% n2nu1
for i=1:r_m
    for j=1:c_m
        if ts(i,j)<=taw || taw==0
                                    if T_M(i,j)~=0
        %n1sum_log_ts=n1sum_log_ts+log(ts(i,j));
        n1sum_log_ts2=n1sum_log_ts2+(log(ts(i,j))-n1nu1)^2;
        n1sum_log_ts3=n1sum_log_ts3+(log(ts(i,j))-n1nu1)^3;
                                    end
        %n1num_sum=n1num_sum+1;
        elseif ts(i,j)>taw
                                    if T_M(i,j)~=0
        %n2sum_log_ts=n2sum_log_ts+log(ts(i,j));
        n2sum_log_ts2=n2sum_log_ts2+(log(ts(i,j))-n2nu1)^2;
        n2sum_log_ts3=n2sum_log_ts3+(log(ts(i,j))-n2nu1)^3;
                                    end
        %n2num_sum=n2num_sum+1;
        end
    end
end 
if n1num_sum ~=0
%n1nu2=n1sum_log_ts2/(n1num_sum-1);
n1nu2=n1sum_log_ts2/(n1num_sum);
%n1nu3=(n1num_sum*n1sum_log_ts3)/(n1num_sum-1)*(n1num_sum-2);
n1nu3=n1sum_log_ts3/(n1num_sum);
n2nu2=n1nu2;
n2nu3=n1nu3;
% n1nu2
% n1nu3
end
if ~(taw==0 || taw==maxts) 
%n2nu2=n2sum_log_ts2/(n2num_sum-1);
n2nu2=n2sum_log_ts2/(n2num_sum);
%n2nu3=(n2num_sum*n2sum_log_ts3)/(n2num_sum-1)*(n2num_sum-2);
n2nu3=n2sum_log_ts3/(n2num_sum);
end

% n2nu2
% n2nu3

       init = fzero(@(x) (psi(1,x*x) - 0.5*abs(n1m1)),1);
       init= abs(init * init);
       p0=zeros(1,2);
       p0(1,1)=50*init;
       p0(1,2)=40*init;
       
%nonlinear LQ by lsnonlin function
       options = optimset('Display','off');
       lb =[0;0]; %Vector of lower bounds
       ub=[1000;100000000]; %Vector of upper bounds
       x=lsqnonlin(@logcum_KI1,p0,lb,ub,options);
       n1alpha=x(1);
       n1lambda=x(2);
       n1mu=n1m1;
       %n1mu=(n1alpha*(exp(2*n1nu1-psi(n1alpha)-psi(n1alpha))))/n1lambda
       %mu=mom1;











% save('n1nu.mat','n1nu2','n1nu3')
% 
% x10=[5,2];
% xx1=fsolve(@logcum1,x10);
% n1alpha = xx1(1);
% n1lambda = xx1(2);

% % 
% % % fixed point method
% % % set up the iteration
% % error1 = 1.e8;
% xx = 2; % initial guesses
% iter=0;
% maxiter=2;
% % begin iteration
% while iter<maxiter
% iter=iter+1;
% syms yy
% s=solve((1/xx)+(1/(2*(xx^2)))+(1/yy)+(1/(2*(yy^2)))==n2nu2);
% rs=real(s);
% yy=rs(1);
% if yy<0
%     yy=rs(2);
% end
% syms xx1
% ss=solve(((-1)/(xx1^2))-(1/(xx1^3))-(((-1)/(yy^2))-(1/(yy^3)))==n2nu3);
% rss=real(ss);
% xx=rss(1);
% if xx<0
%     xx=rss(2);
% end
% % % calculate norm
% % error1=sqrt(Y(1)^2+Y(2)^2);
% % 
% end
% n1alpha = xx;
% n1lambda = yy;
% 

% % Newton Raphson solution of two nonlinear algebraic equations
% % set up the iteration
% error1 = 1.e8;
% xx(1) = .01; % initial guesses of alpha
% xx(2) = .02;%lambda
% iter=0;
% maxiter=15;
% % begin iteration
% while error1>.01 && iter<maxiter
%     if xx(1)<0 || xx(2)<0
%     msgbox('This is not suitable method for parameter estimation!','not suitable PE method','error');
%     return
%     end
% iter=iter+1;
% x = xx(1);
% y = xx(2);
% % calculate the functions
% f(1) = psi(1,x)+psi(1,y)-4*n1nu2;
% f(2) = psi(2,x)-psi(2,y)-8*n1nu3;
% % calculate the Jacobian
% J1(1,1) = psi(2,x);
% J1(1,2) = psi(2,y);
% J1(2,1) = psi(3,x);
% J1(2,2) = -psi(3,y);
% 
% % solve the linear equations
% Y = -J1\f';
% % move the solution, xx(k+1) - xx(k), to xx(k+1)
% xx = xx + Y';
% 
% % calculate norm
% error1=sqrt(Y(1)^2+Y(2)^2);
% 
% end
% n1alpha = xx1(1);
% n1lambda = xx1(2);

% %Non-linear least square method
% l_o=[n1nu2;n1nu3];
% maxiter=20;
% x=[5;13];
% for iter=1:maxiter
% xx=x(1,1);
% yy=x(2,1);
% l_c=[(psi(1,xx)+psi(1,yy));(psi(2,xx)-psi(2,yy))];
% A=[psi(2,xx) psi(2,yy);psi(3,xx) -psi(3,yy)];
% delta_l=l_o - l_c;
% delta_x=((A'*A)\A')*delta_l;
% error=sqrt(delta_x(1)^2+delta_x(2)^2);
% if error<0.01
%     break
% end
% x=abs(x+delta_x);
% end
% n1alpha=x(1);
% n1lambda=x(2);
% 
% 
% 
% n1mu=(n1alpha*(10^(2*n1nu1-psi(n1alpha)-psi(n1alpha))))/n1lambda;

if ~(taw==0 || taw==maxts) 


    
    
    
       init = fzero(@(x) (psi(1,x*x) - 0.5*abs(n2m1)),1);
       init= abs(init * init);
       p0=zeros(1,2);
       p0(1,1)=50*init;
       p0(1,2)=40*init;
       
%nonlinear LQ by lsnonlin function
       options = optimset('Display','off');
       lb =[0;0]; %Vector of lower bounds
       ub=[1000;100000000]; %Vector of upper bounds
       x=lsqnonlin(@logcum_KI2,p0,lb,ub,options);
       n2alpha=x(1);
       n2lambda=x(2);
       n2mu=n2m1;
       %n2mu=(n2alpha*(exp(2*n2nu1-psi(n2alpha)-psi(n2alpha))))/n2lambda
       %mu=mom1;

   
    
    
    
%x20=[5,2];
%save('n2nu.mat','n2nu2','n2nu3')
%xx2=fsolve(@logcum2,x20);
% n2alpha = xx2(1);
% n2lambda = xx2(2);



% % Newton Raphson solution of two nonlinear algebraic equations
% % set up the iteration
% error1 = 1.e8;
% xx(1) = .01; % initial guesses
% xx(2) = .02;
% iter=0;
% maxiter=15;
% % begin iteration
% while error1>.01 && iter<maxiter
%     if xx(1)<0 || xx(2)<0
%     msgbox('This is not suitable method for parameter estimation!','not suitable PE method','error');
%     return
%     end
% iter=iter+1;
% x = xx(1);
% y = xx(2);
% % calculate the functions
% f(1) = psi(1,x)+psi(1,y)-4*n2nu2;
% f(2) = psi(2,x)-psi(2,y)-8*n2nu3;
% % calculate the Jacobian
% J1(1,1) = psi(2,x);
% J1(1,2) = psi(2,y);
% J1(2,1) = psi(3,x);
% J1(2,2) = -psi(3,y);
% % solve the linear equations
% Y= -J1\f';
% % move the solution, xx(k+1) - xx(k), to xx(k+1)
% xx = xx + Y';
% 
% % calculate norm
% error1=sqrt(Y(1)^2+Y(2)^2);
% 
% end
% n2alpha = xx2(1);
% n2lambda = xx2(2);


% 
% % fixed point method
% % set up the iteration
% error1 = 1.e8;
% x = 1; % initial guesses
% iter=0;
% maxiter=2;
% begin iteration
% while iter<maxiter
% iter=iter+1;
% syms y
% s=solve((1/x)+(1/(2*(x^2)))+(1/y)+(1/(2*(y^2)))==n2nu2);
% rs=real(s);
% y=rs(1);
% if y<0
%     y=rs(2);
% end
% syms x
% ss=solve((-1/x^2)-(1/(x^3))-((-1/y^2)-(1/(y^3)))==n2nu3);
% rss=real(ss);
% x=rss(1);
% if x<0
%     x=rss(2);
% end
% % calculate norm
% error1=sqrt(Y(1)^2+Y(2)^2);
% 
% end
% n2alpha = x;
% n2lambda = y;


% %Non-linear least square method
% l_o=[n2nu2;n2nu3];
% maxiter=20;
% x=[15;3];
% for iter=1:maxiter
% xx=x(1,1);
% yy=x(2,1);
% l_c=[(psi(1,xx)+psi(1,yy));(psi(2,xx)-psi(2,yy))];
% A=[psi(2,xx) psi(2,yy);psi(3,xx) -psi(3,yy)];
% delta_l=l_o - l_c;
% delta_x=((A'*A)\A')*delta_l;
% error=sqrt(delta_x(1)^2+delta_x(2)^2);
% if error<0.01
%     break
% end
% x=abs(x+delta_x);
% end
% n2alpha=x(1);
% n2lambda=x(2);
% 
% 
% n2mu=(n2alpha*(10^(2*n2nu1-psi(n2alpha)-psi(n2alpha))))/n2lambda;

end

if taw==0 || taw==maxts 
n2alpha=n1alpha;
n2lambda=n1lambda;
n2mu=n1mu;
end


elseif val_pdf(2)==1
    %% PE of Gamma pdf by log cumulant method 
    
n1sum_ts=0;
n2sum_ts=0;
n1sum_log_ts=0;%This parameter is equal to summation of log(ts).(class n1) 
n1sum_log_ts2=0;%This parameter is equal to (summation of (log(ts)^2) minus nu1^2.
%n1sum_log_ts3=0;%This parameter is equal to summation of (log(ts)-n1nu1)^3.
n1num_sum=0;
n2sum_log_ts=0;%This parameter is equal to summation of log(ts). (class n2)
n2sum_log_ts2=0;%This parameter is equal to summation of (log(ts)-n2nu1)^2.
%n2sum_log_ts3=0;%This parameter is equal to summation of (log(ts)-n2nu1^3.
n2num_sum=0;
for i=1:r_m
    for j=1:c_m
        if ts(i,j)<=taw || taw==0
                        if T_M(i,j)~=0
        n1sum_log_ts=n1sum_log_ts+log(ts(i,j));
        n1sum_ts=n1sum_ts+ts(i,j);
                        end
        n1num_sum=n1num_sum+1;
        elseif ts(i,j)>taw
                                    if T_M(i,j)~=0
        n2sum_log_ts=n2sum_log_ts+log(ts(i,j));
        n2sum_ts=n2sum_ts+ts(i,j);
                                    end
        n2num_sum=n2num_sum+1;
        end
    end
end

if n1num_sum ~=0
n1m1=n1sum_ts/n1num_sum;
n2m1=n1m1;
n1nu11=n1sum_log_ts/n1num_sum;

n1nu1=n1nu11;
n2nu11=n1nu11;
n2nu1=n2nu11;
end
if ~(taw==0 || taw==maxts) 
n2m1=n2sum_ts/n2num_sum;
n2nu11=n2sum_log_ts/n2num_sum;
n2nu1=n2nu11;
end

for i=1:r_m
    for j=1:c_m
        if ts(i,j)<=taw || taw==0
                                    if T_M(i,j)~=0
        n1sum_log_ts2=n1sum_log_ts2+(log(ts(i,j))-n1nu1)^2;
        %n1sum_log_ts3=n1sum_log_ts3+(log(ts(i,j))-n1nu1)^3;
                                   end
        elseif ts(i,j)>taw
                                    if T_M(i,j)~=0
        %n2sum_log_ts=n2sum_log_ts+log(ts(i,j));
        n2sum_log_ts2=n2sum_log_ts2+(log(ts(i,j))-n2nu1)^2;
        %n2sum_log_ts3=n2sum_log_ts3+(log(ts(i,j))-n2nu1)^3;
                                    end
        end
    end
end 
if n1num_sum ~=0
n1nu2=n1sum_log_ts2/(n1num_sum);
%n1nu3=n1sum_log_ts3/(n1num_sum);
n2nu2=n1nu2;
%n2nu3=n1nu3;
end
if ~(taw==0 || taw==maxts) 
n2nu2=n2sum_log_ts2/(n2num_sum);
%n2nu3=n2sum_log_ts3/(n2num_sum);
end

       i0=1;
%nonlinear LQ by lsnonlin function
       options = optimset('Display','off');
       lb =0; %Vector of lower bounds
       ub=100000000; %Vector of upper bounds
       x=lsqnonlin(@n1kGamma,i0,lb,ub,options);
       n1l=x;
n1mu=exp(n1nu1+log(n1l)-psi(n1l));       
% 
% 
% 
% 
% % Newton Raphson solution of two nonlinear algebraic equations
% % set up the iteration
% error1 = 1.e8;
% xx(1) = .5; % initial guesses of mu
% xx(2) = .045;%l
% iter=0;
% maxiter=50;
% % begin iteration
% while error1>.1 && iter<maxiter
%     if xx(2)<0 || xx(1)/xx(2)<0
%     msgbox('This is not suitable method for parameter estimation!','not suitable PE method','error');
%     return
%     end
% iter=iter+1;
% x = xx(1);
% y = xx(2);
% % calculate the functions
% f(1) = log10(x/y)+psi(y)-n1nu1;
% f(2) = psi(1,y)-n1nu2;
% % calculate the Jacobian
% J1(1,1) = 1/x;
% J1(1,2) = (-1/y)+psi(1,y);
% J1(2,1) = 0;
% J1(2,2) = psi(2,y);
% 
% % solve the linear equations
% Y = -J1\f';
% % move the solution, xx(k+1) - xx(k), to xx(k+1)
% xx = xx + Y';
% 
% % calculate norms
% error1=sqrt(Y(1)^2+Y(2)^2);
% 
% end
% 
% n1mu = xx(1);
% n1l = xx(2);
% 

if ~(taw==0 || taw==maxts) 

    
    
           j0=1;
%nonlinear LQ by lsnonlin function
       options = optimset('Display','off');
       lb =0; %Vector of lower bounds
       ub=100000000; %Vector of upper bounds
       y=lsqnonlin(@n2kGamma,j0,lb,ub,options);
       n2l=y;
    n2mu=exp(n2nu1+log(n2l)-psi(n2l));
%     
%     
%     
%     
% % Newton Raphson solution of two nonlinear algebraic equations
% % set up the iteration
% error1 = 1.e8;
% xx(1) = .5; % initial guesses
% xx(2) = .045;
% iter=0;
% maxiter=50;
% % begin iteration
% while error1>.1 && iter<maxiter
% iter=iter+1;
% x = xx(1);
% y = xx(2);
% % calculate the functions
% f(1) = log10(x/y)+psi(y)-n2nu1;
% f(2) = psi(1,y)-n2nu2;
% % calculate the Jacobian
% J1(1,1) = 1/x;
% J1(1,2) = (-1/y)+psi(1,y);
% J1(2,1) = 0;
% J1(2,2) = psi(2,y);
% % solve the linear equations
% Y= -J1\f';
% % move the solution, xx(k+1) - xx(k), to xx(k+1)
% xx = xx + Y';
% 
% % calculate norms
% error1=sqrt(Y(1)^2+Y(2)^2);
% 
% end
% 
% n2mu = xx(1);
% n2l = xx(2);
end

if taw==0 || taw==maxts 
n2mu=n1mu;
n2l=n1l;

end


elseif val_pdf(3)==1

    %% PE for LogNormal pdf
% 
% n1sum_ts=0;
% n2sum_ts=0;
% n1sum_log_ts=0;%This parameter is equal to summation of log(ts).(class n1) 
% n1sum_log_ts2=0;%This parameter is equal to (summation of (log(ts)^2) minus nu1^2.
% n1sum_log_ts3=0;%This parameter is equal to summation of (log(ts)-n1nu1)^3.
% n1num_sum=0;
% n2sum_log_ts=0;%This parameter is equal to summation of log(ts). (class n2)
% n2sum_log_ts2=0;%This parameter is equal to summation of (log(ts)-n2nu1)^2.
% n2sum_log_ts3=0;%This parameter is equal to summation of (log(ts)-n2nu1^3.
% n2num_sum=0;
% for i=1:r_m
%     for j=1:c_m
%         if ts(i,j)<=taw || taw==0
%         n1sum_log_ts=n1sum_log_ts+log(ts(i,j));
%         n1sum_ts=n1sum_ts+ts(i,j);
%         n1sum_log_ts2=n1sum_log_ts2+(log10(ts(i,j)))^2;
%         n1sum_log_ts3=n1sum_log_ts3+(log10(ts(i,j)))^3;
%         n1num_sum=n1num_sum+1;
%         elseif ts(i,j)>taw
%         n2sum_log_ts=n2sum_log_ts+log(ts(i,j));
%         n2sum_ts=n2sum_ts+ts(i,j);
%         n2sum_log_ts2=n2sum_log_ts2+(log10(ts(i,j)))^2;
%         n2sum_log_ts3=n2sum_log_ts3+(log10(ts(i,j)))^3;
%         n2num_sum=n2num_sum+1;
%         end
%     end
% end
% n1sum_ts
% n1num_sum
% n1m1
% n2sum_ts
% n2num_sum
% if n1num_sum ~=0
% n1m1=n1sum_ts/n1num_sum;
% n2m1=n1m1;
% n1nu11=n1sum_log_ts/n1num_sum;
% n1nu1=n1nu11;
% n2nu11=n1nu11;
% n2nu1=n2nu11;
% else
%     n1nu1=1;
%     n2nu1=5;
% end
% if ~(taw==0 || taw==maxts) 
% n2m1=n2sum_ts/n2num_sum;
% n2nu11=n2sum_log_ts/n2num_sum;
% n2nu1=n2nu11;
% end
% n1nu1
% n2m1
% n2nu1
% for i=1:r_m
%     for j=1:c_m
%         if ts(i,j)<=taw || taw==0
%         n1sum_log_ts=n1sum_log_ts+log(ts(i,j));
%         n1sum_log_ts2=n1sum_log_ts2+(log(ts(i,j))-n1nu1)^2;
%         n1sum_log_ts3=n1sum_log_ts3+(log(ts(i,j))-n1nu1)^3;
%         n1num_sum=n1num_sum+1;
%         elseif ts(i,j)>taw
%         n2sum_log_ts=n2sum_log_ts+log(ts(i,j));
%         n2sum_log_ts2=n2sum_log_ts2+(log(ts(i,j))-n2nu1)^2;
%         n2sum_log_ts3=n2sum_log_ts3+(log(ts(i,j))-n2nu1)^3;
%         n2num_sum=n2num_sum+1;
%         end
%     end
% end 
% if n1num_sum ~=0
% n1nu2=n1sum_log_ts2/(n1num_sum-1);
% n1nu2=n1sum_log_ts2/(n1num_sum);
% n1nu3=(n1num_sum*n1sum_log_ts3)/(n1num_sum-1)*(n1num_sum-2);
% n1nu3=n1sum_log_ts3/(n1num_sum);
% n2nu2=n1nu2;
% n2nu3=n1nu3;
% n1nu2
% n1nu3
% else
%     n1nu2=.1;
%     n2nu2=.1;
% end
% if ~(taw==0 || taw==maxts) 
% n2nu2=n2sum_log_ts2/(n2num_sum-1);
% n2nu2=n2sum_log_ts2/(n2num_sum);
% n2nu3=(n2num_sum*n2sum_log_ts3)/(n2num_sum-1)*(n2num_sum-2);
% n2nu3=n2sum_log_ts3/(n2num_sum);
% end
% n1phi=n1nu1
% n2phi=n2nu1
% n1kisi=n1nu2
% n2kisi=n2nu2
    
n1sum_hlog=0;%This parameter is equal to summation of h(q)log(q).(class n1) 
n1num_sum=0;
n2sum_hlog=0;%This parameter is equal to summation of h(w)log(w). (class n2)
n2num_sum=0;
for q=0:taw
    if q==0
       n1sum_hlog=n1sum_hlog+h(q+1)*log(.1);
    else
       n1sum_hlog=n1sum_hlog+h(q+1)*log(q);
    end
    n1num_sum=n1num_sum+h(q+1);
end
for w=taw+1:maxts
    if w==0
      n2sum_hlog=n2sum_hlog+h(w+1)*log(.1);
    else
      n2sum_hlog=n2sum_hlog+h(w+1)*log(w);
    end
    n2num_sum=n2num_sum+h(w+1); 
end
if n1num_sum ~=0
n1phi=n1sum_hlog/n1num_sum;
n2phi=n2sum_hlog/n2num_sum;
else
    n1phi=1;
    n2phi=1;
end
n1sum_hlog2=0;%This parameter is equal to summation of h(q)log(q).(class n1) 
n2sum_hlog2=0;%This parameter is equal to summation of h(w)log(w). (class n2)
for q=0:taw
    if q==0
       n1sum_hlog2=n1sum_hlog2+h(q+1)*(log(.1)-(n1phi.^2));
    else
       n1sum_hlog2=n1sum_hlog2+h(q+1)*(log(q)-(n1phi.^2));
    end
end
for w=taw+1:maxts
    if w==0
      n2sum_hlog2=n2sum_hlog2+h(w+1)*(log(.1)-(n2phi.^2));
    else
      n2sum_hlog2=n2sum_hlog2+h(w+1)*(log(w)-(n2phi.^2));
    end
end
if n1num_sum ~=0
n1kisi=n1sum_hlog2/n1num_sum;
n2kisi=n2sum_hlog2/n2num_sum;
else
    n1kisi=1;
    n2kisi=1;
end
%  n1phi
%  n2phi
%  n1kisi
%  n2kisi
    
elseif val_pdf(4)==1
    %% PE:Weibull ratio-Log cumulant

n1sum_hlog=0;%This parameter is equal to summation of h(q)log(q).(class n1) 
n1num_sum=0;
n2sum_hlog=0;%This parameter is equal to summation of h(w)log(w). (class n2)
n2num_sum=0;
for q=0:taw
    if q==0
       n1sum_hlog=n1sum_hlog+h(q+1)*log(.1);
    else
       n1sum_hlog=n1sum_hlog+h(q+1)*log(q);
    end
    n1num_sum=n1num_sum+h(q+1);
end
for w=taw+1:maxts
    if w==0
      n2sum_hlog=n2sum_hlog+h(w+1)*log(.1);
    else
      n2sum_hlog=n2sum_hlog+h(w+1)*log(w);
    end
    n2num_sum=n2num_sum+h(w+1); 
end
if n1num_sum ~=0
n1phi=n1sum_hlog/n1num_sum;
n2phi=n2sum_hlog/n2num_sum;
else
    n1phi=10;
    n2phi=20;
end
n1sum_hlog2=0;%This parameter is equal to summation of h(q)log(q).(class n1) 
n2sum_hlog2=0;%This parameter is equal to summation of h(w)log(w). (class n2)
for q=0:taw
    if q==0
       n1sum_hlog2=n1sum_hlog2+h(q+1)*(log(.1)-(n1phi.^2));
    else
       n1sum_hlog2=n1sum_hlog2+h(q+1)*(log(q)-(n1phi.^2));
    end
end
for w=taw+1:maxts
    if w==0
      n2sum_hlog2=n2sum_hlog2+h(w+1)*(log(.1)-(n2phi.^2));
    else
      n2sum_hlog2=n2sum_hlog2+h(w+1)*(log(w)-(n2phi.^2));
    end
end
if n1num_sum ~=0
n1kisi=n1sum_hlog2/n1num_sum;
n2kisi=n2sum_hlog2/n2num_sum;
else
    n1kisi=1;
    n2kisi=1;
end
n1lambda=exp(n1phi);
n2lambda=exp(n2phi);
n1eta=sqrt(2*psi(1,1)/n1kisi);
n2eta=sqrt(2*psi(1,1)/n2kisi);


elseif val_pdf(5)==1

    %% PE for Generalized Gamma pdf 
    
n1sum_ts=0;
n2sum_ts=0;
n1sum_log_ts=0;%This parameter is equal to summation of log(ts).(class n1) 
n1sum_log_ts2=0;%This parameter is equal to (summation of (log(ts)^2) minus nu1^2.
n1sum_log_ts3=0;%This parameter is equal to summation of (log(ts)-n1nu1)^3.
n1num_sum=0;
n2sum_log_ts=0;%This parameter is equal to summation of log(ts). (class n2)
n2sum_log_ts2=0;%This parameter is equal to summation of (log(ts)-n2nu1)^2.
n2sum_log_ts3=0;%This parameter is equal to summation of (log(ts)-n2nu1^3.
n2num_sum=0;
for i=1:r_m
    for j=1:c_m
        if ts(i,j)<=taw || taw==0
            if T_M(i,j)~=0
        n1sum_log_ts=n1sum_log_ts+log(ts(i,j));
        n1sum_ts=n1sum_ts+ts(i,j);

        n1num_sum=n1num_sum+1;
            end
        elseif ts(i,j)>taw
            if T_M(i,j)~=0
        n2sum_log_ts=n2sum_log_ts+log(ts(i,j));
        n2sum_ts=n2sum_ts+ts(i,j);

        n2num_sum=n2num_sum+1;
            end
        end
    end
end

if n1num_sum ~=0
n1m1=n1sum_ts/n1num_sum;
n2m1=n1m1;
n1nu11=n1sum_log_ts/n1num_sum;

n1nu1=n1nu11;
n2nu11=n1nu11;
n2nu1=n2nu11;
end
if ~(taw==0 || taw==maxts) 
n2m1=n2sum_ts/n2num_sum;
n2nu11=n2sum_log_ts/n2num_sum;
n2nu1=n2nu11;
end

for i=1:r_m
    for j=1:c_m
        if ts(i,j)<=taw || taw==0
                        if T_M(i,j)~=0
        n1sum_log_ts2=n1sum_log_ts2+(log(ts(i,j))-n1nu1)^2;
        n1sum_log_ts3=n1sum_log_ts3+(log(ts(i,j))-n1nu1)^3;
                        end
        elseif ts(i,j)>taw
                        if T_M(i,j)~=0
        %n2sum_log_ts=n2sum_log_ts+log(ts(i,j));
        n2sum_log_ts2=n2sum_log_ts2+(log(ts(i,j))-n2nu1)^2;
        n2sum_log_ts3=n2sum_log_ts3+(log(ts(i,j))-n2nu1)^3;
                        end
        end
    end
end 
if n1num_sum ~=0
n1nu2=n1sum_log_ts2/(n1num_sum);
n1nu3=n1sum_log_ts3/(n1num_sum);
n2nu2=n1nu2;
n2nu3=n1nu3;
end
if ~(taw==0 || taw==maxts) 
n2nu2=n2sum_log_ts2/(n2num_sum);
n2nu3=n2sum_log_ts3/(n2num_sum);
end
       i0=-1;
%nonlinear LQ by lsnonlin function
       options = optimset('Display','off');
       lb =0; %Vector of lower bounds
       ub=100000000; %Vector of upper bounds
       x=lsqnonlin(@n1kGGamma,i0,lb,ub,options);
       n1k=x;
       
       j0=-1;
%nonlinear LQ by lsnonlin function
       options = optimset('Display','off');
       lb =0; %Vector of lower bounds
       ub=100000000; %Vector of upper bounds
       y=lsqnonlin(@n2kGGamma,j0,lb,ub,options);
       n2k=y;
% syms n1K
% n1S = solve((n1nu2^3/n1nu3^2)== (psi(1,n1K)^3)/(psi(2,n1K)^2),n1K);
% n1SS=eval(n1S);
% n1k=n1SS(1,1)
% n2nu2
% n2nu3
% pause
% syms n2K
% n2S = solve((n2nu2^3/n2nu3^2)== (psi(1,n2K)^3)/(psi(2,n2K)^2),n2K);
% n2SS=eval(n2S);
% n2k=n2SS(1,1);

n1v=sign(-n1nu3)*sqrt(psi(1,n1k)/n1nu2);
n1sig=exp(n1nu1-(psi(n1k)/n1v));
n2v=sign(-n2nu3)*sqrt(psi(1,n2k)/n2nu2);
n2sig=exp(n2nu1-(psi(n2k)/n2v));


end

%mixed method


elseif value1_1(3)==1


if val_pdf(1)==1    
    %% PE of fisher pdf by mixed method
    
n1sum_ts=0;%This parameter is equal to summation of (ts). 
n1sum_ts2=0;%This parameter is equal to summation of(ts)^2.
n1sums_log_ts0=0;%This parameter is equal to summation of log(ts). 
n1sums_log_ts1=0;%This parameter is equal to summation of s*log(ts).
n1sums_log_ts2=0;%This parameter is equal to summation of s^2*log(ts).
n1num_sum=0;
n2sum_ts=0;%This parameter is equal to summation of (ts). 
n2sum_ts2=0;%This parameter is equal to summation of(ts)^2.
n2sums_log_ts0=0;%This parameter is equal to summation of log(ts). 
n2sums_log_ts1=0;%This parameter is equal to summation of s*log(ts).
n2sums_log_ts2=0;%This parameter is equal to summation of s^2*log(ts).
n2num_sum=0;

for i=1:r_m
    for j=1:c_m
        if ts(i,j)<=taw || taw==0
        n1sum_ts=n1sum_ts+(ts(i,j));
        n1sum_ts2=n1sum_ts2+((ts(i,j)))^2;
        n1sums_log_ts0=n1sums_log_ts0+log(ts(i,j));
        n1sums_log_ts1=n1sums_log_ts1+((ts(i,j))*log(ts(i,j)));
        n1sums_log_ts2=n1sums_log_ts2+(((ts(i,j))^2)*log(ts(i,j)));
        n1num_sum=n1num_sum+1;
        elseif ts(i,j)>taw
        n2sum_ts=n2sum_ts+(ts(i,j));
        n2sum_ts2=n2sum_ts2+((ts(i,j)))^2;
        n2sums_log_ts0=n2sums_log_ts0+log(ts(i,j));
        n2sums_log_ts1=n2sums_log_ts1+((ts(i,j))*log(ts(i,j)));
        n2sums_log_ts2=n2sums_log_ts2+(((ts(i,j))^2)*log(ts(i,j)));
        n2num_sum=n2num_sum+1;
        end
    end
end
if n1num_sum~=0
n1m1=n1sum_ts/n1num_sum;
n1m2=n1sum_ts2/n1num_sum;
n1w0=n1sums_log_ts0/n1num_sum;
n1w1=n1sums_log_ts1/n1num_sum;
n1w2=n1sums_log_ts2/n1num_sum;
end
if ~(taw==0 || taw==maxts) 
n2m1=n2sum_ts/n2num_sum;
n2m2=n2sum_ts2/n2num_sum;
n2w0=n2sums_log_ts0/n2num_sum;
n2w1=n2sums_log_ts1/n2num_sum;
n2w2=n2sums_log_ts2/n2num_sum;
end

%  
%        init = fzero(@(x) (psi(1,x*x) - 0.5*abs(n1m1)),1);
%        init= abs(init * init);
%        p0=zeros(1,3);
%        p0(1,1)=50*init;
%        p0(1,2)=40*init;
%        p0(1,3)=n1m1;
% %nonlinear LQ by lsnonlin function
%        options = optimset('Display','off');
%        lb =[0;0]; %Vector of lower bounds
%        ub=[1000;100000000]; %Vector of upper bounds
%        x=lsqnonlin(@mixedPE_KI1,p0,lb,ub,options);
%        n1alpha=x(1);
%        n1lambda=x(2);
%        n1mu=x(3);
%        %mu=mom1;
% 
% 







syms n1a n1l n1m
[n1aa,n1ll,n1mm] = solve(n1m1== n1l*n1m/(n1l-1), (n1w1/n1m1)-n1w0==(1/n1a)+(1/(n1l-1)), (n1w2/n1m2)-(n1w1/n1m1)==(1/(n1a+1))+(1/(n1l-2)),n1a,n1l,n1m);
n1a=eval(n1aa);
n1l=eval(n1ll);
n1m=eval(n1mm);
            
n1alpha=max(n1a);
n1lambda=max(n1l);
n1mu=max(n1m);

if ~(taw==0 || taw==255) 
    
%     
%     
%        init = fzero(@(x) (psi(1,x*x) - 0.5*abs(n2m1)),1);
%        init= abs(init * init);
%        p0=zeros(1,3);
%        p0(1,1)=50*init;
%        p0(1,2)=40*init;
%        p0(1,3)=n2m1;
% %nonlinear LQ by lsnonlin function
%        options = optimset('Display','off');
%        lb =[0;0]; %Vector of lower bounds
%        ub=[1000;100000000]; %Vector of upper bounds
%        x=lsqnonlin(@mixedPE_KI2,p0,lb,ub,options);
%        n2alpha=x(1);
%        n2lambda=x(2);
%        n2mu=x(3);
%        %mu=mom1;
%     
%     

syms n2a n2l n2m
[n2aa,n2ll,n2mm] = solve(n2m1== n2l*n2m/(n2l-1), (n2w1/n2m1)-n2w0==(1/n2a)+(1/(n2l-1)), (n2w2/n2m2)-(n2w1/n2m1)==(1/(n2a+1))+(1/(n2l-2)),n2a,n2l,n2m);
n2a=eval(n2aa);
n2l=eval(n2ll);
n2m=eval(n2mm);

n2alpha=max(n2a);
n2lambda=max(n2l);
n2mu=max(n2m);
end
if taw==0 || taw==maxts 
n2alpha=n1alpha;
n2lambda=n1lambda;
n2mu=n1mu;
end


elseif val_pdf(2)==1
    %% PE of Gamma pdf by mixed method
    
n1sum_ts=0;%This parameter is equal to summation of (ts). 
n1sum_ts2=0;%This parameter is equal to summation of(ts)^2.
n1sums_log_ts0=0;%This parameter is equal to summation of log(ts). 
n1sums_log_ts1=0;%This parameter is equal to summation of s*log(ts).
n1sums_log_ts2=0;%This parameter is equal to summation of s^2*log(ts).
n1num_sum=0;
n2sum_ts=0;%This parameter is equal to summation of (ts). 
n2sum_ts2=0;%This parameter is equal to summation of(ts)^2.
n2sums_log_ts0=0;%This parameter is equal to summation of log(ts). 
n2sums_log_ts1=0;%This parameter is equal to summation of s*log(ts).
n2sums_log_ts2=0;%This parameter is equal to summation of s^2*log(ts).
n2num_sum=0;
for i=1:r_m
    for j=1:c_m
        if ts(i,j)<=taw || taw==0
        n1sums_log_ts0=n1sums_log_ts0+log10(ts(i,j));
        n1sums_log_ts1=n1sums_log_ts1+((ts(i,j))*log10(ts(i,j)));
        n1num_sum=n1num_sum+1;
        elseif ts(i,j)>taw
        n2sums_log_ts0=n2sums_log_ts0+log10(ts(i,j));
        n2sums_log_ts1=n2sums_log_ts1+((ts(i,j))*log10(ts(i,j)));
        n2num_sum=n2num_sum+1;
        end
    end
end

n1w0=n1sums_log_ts0/n1num_sum;
n1w1=n1sums_log_ts1/n1num_sum;

if ~(taw==0 || taw==255) 

n2w0=n2sums_log_ts0/n2num_sum;
n2w1=n2sums_log_ts1/n2num_sum;

end
syms n1mu n1l
n1S = solve(n1w0== psi(n1l), n1w1==(psi(n1l+1)-(n1mu/n1l))/((gamma(n1l)*n1l)/gamma(n1l+1)*n1mu));
n1S=eval(subs([n1S.n1mu n1S.n1l]));
n1mu=abs(n1S(1,1));
n1l=abs(n1S(1,2));

if ~(taw==0 || taw==255) 
syms n2mu n2l
n2S = solve(n2w0== psi(n2l), n2w1==(psi(n2l+1)-(n2mu/n2l))/((gamma(n2l)*n1l)/gamma(n2l+1)*n2mu));
n2S=eval(subs([n2S.n2mu n2S.n2l]));
n2mu=abs(n2S(1,1));
n2l=abs(n2S(1,2));

end
if taw==0 || taw==255 
n2mu=n1mu;
n2l=n1l;

end
    if n1S(2)<0 || n1S(2)<0
    msgbox('This is not suitable method for parameter estimation!','not suitable PE method','error');
    break
    end
    if n2S(1)<0 || n2S(2)<0
    msgbox('This is not suitable method for parameter estimation!','not suitable PE method','error');
    break
    end
end
end

xnc=zeros(taw+1,1);%no change class values
fnc=zeros(taw+1,1);%no change class histogram values
xc=zeros(maxts-taw,1);%change class values
fc=zeros(maxts-taw,1);%change class histogram values
            P_0_taw=0;
            H0_2ndterm=0;
            
            if val_pdf(1)==1
                %% KI main body for Fisher pdf
                
            iii=0;%numerator            
            for q=0:taw
                P_0_taw=P_0_taw+h(q+1);
                if q~=0
                iii=iii+1;
                xnc(iii,1)=q;
                fnc(iii,1)=h2(q+1);
                xc(iii,1)=q;
                fc(iii,1)=h3(q+1);
                H0_2ndterm=H0_2ndterm+h(q+1).*(log(n1alpha)+(n1alpha-1).*log(n1alpha*q/n1lambda*n1mu)-...
                          log(n1lambda)-log(n1mu)-(n1alpha+n1lambda).*log(1+n1alpha*q/n1lambda*n1mu)...
                              -gammaln(n1alpha)-gammaln(n1lambda)+gammaln(n1alpha+n1lambda));                
                end
                if q==0%because of that our quantized data is: .1 1 2 3 ... 255     
                iii=iii+1;
                xnc(iii,1)=0.1;
                fnc(iii,1)=h2(q+1);
                xc(iii,1)=0.1;
                fc(iii,1)=h3(q+1);
                H0_2ndterm=0;
                H0_2ndterm=H0_2ndterm+h(q+1).*(log(n1alpha)+(n1alpha-1).*log(n1alpha*.1/n1lambda*n1mu)-...
                          log(n1lambda)-log(n1mu)-(n1alpha+n1lambda).*log(1+n1alpha*.1/n1lambda*n1mu)...
                              -gammaln(n1alpha)-gammaln(n1lambda)+gammaln(n1alpha+n1lambda));                
                end
            end

            %up to plot pdfs
            npnc=zeros(maxts+1,1);
            fnpnc=zeros(maxts+1,1);
            npc=zeros((maxts+1),1);
            fnpc=zeros((maxts+1),1);
            id=0;
            for npnc1=0:taw
                id=id+1;
                if npnc1==0
                npnc(id)=.1;
                npc(id)=.1;
                fnpnc(id)=exp(log(n1alpha)+(n1alpha-1).*log(n1alpha*.1/n1lambda*n1mu)-...
                          log(n1lambda)-log(n1mu)-(n1alpha+n1lambda).*log(1+n1alpha*.1/n1lambda*n1mu)...
                              -gammaln(n1alpha)-gammaln(n1lambda)+gammaln(n1alpha+n1lambda));                
                fnpc(id)=exp(log(n2alpha)+(n2alpha-1).*log(n2alpha*.1/n2lambda*n2mu)-...
                          log(n2lambda)-log(n2mu)-(n2alpha+n2lambda).*log(1+n2alpha*.1/n2lambda*n2mu)...
                              -gammaln(n2alpha)-gammaln(n2lambda)+gammaln(n2alpha+n2lambda));

                else
                npnc(id)=npnc1;
                npc(id)=npnc1;                
                fnpnc(id)=exp(log(n1alpha)+(n1alpha-1).*log(n1alpha*npnc1/n1lambda*n1mu)-...
                          log(n1lambda)-log(n1mu)-(n1alpha+n1lambda).*log(1+n1alpha*npnc1/n1lambda*n1mu)...
                              -gammaln(n1alpha)-gammaln(n1lambda)+gammaln(n1alpha+n1lambda));                
                fnpc(id)=exp(log(n2alpha)+(n2alpha-1).*log(n2alpha*npnc1/n2lambda*n2mu)-...
                          log(n2lambda)-log(n2mu)-(n2alpha+n2lambda).*log(1+n2alpha*npnc1/n2lambda*n2mu)...
                              -gammaln(n2alpha)-gammaln(n2lambda)+gammaln(n2alpha+n2lambda));
                end            
            end
            P_1_taw=0;
            H1_2ndterm=0;

            for w=(taw+1):maxts
                P_1_taw=P_1_taw+h(w+1);
                if w~=0

                iii=iii+1;    
                xc(iii,1)=w;
                fc(iii,1)=h3(w+1);
                xnc(iii,1)=w;
                fnc(iii,1)=h2(w+1);
                H1_2ndterm=H1_2ndterm+h(w+1).*(log(n2alpha)+(n2alpha-1).*log(n2alpha*w/n2lambda*n2mu)-...
                          log(n2lambda)-log(n2mu)-(n2alpha+n2lambda).*log(1+n2alpha*w/n2lambda*n2mu)...
                              -gammaln(n2alpha)-gammaln(n2lambda)+gammaln(n2alpha+n2lambda));
                end
                if w==0
                iii=iii+1;    
                xc(iii,1)=0.1;
                fc(iii,1)=h3(w+1);
                xnc(iii,1)=0.1;
                fnc(iii,1)=h2(w+1);                                    
                H1_2ndterm=0;
                H1_2ndterm=H1_2ndterm+h(w+1).*(log(n2alpha)+(n2alpha-1).*log(n2alpha*.1/n2lambda*n2mu)-...
                          log(n2lambda)-log(n2mu)-(n2alpha+n2lambda).*log(1+n2alpha*.1/n2lambda*n2mu)...
                              -gammaln(n2alpha)-gammaln(n2lambda)+gammaln(n2alpha+n2lambda));
                end
            end
            %be up to plot estimated pdfs

            for npc1=(taw+1):maxts
                id=id+1;
                if npc1==0
                npc(id)=.1;
                npnc(id)=.1;
                fnpnc(id)=exp(log(n1alpha)+(n1alpha-1).*log(n1alpha*.1/n1lambda*n1mu)-...
                          log(n1lambda)-log(n1mu)-(n1alpha+n1lambda).*log(1+n1alpha*.1/n1lambda*n1mu)...
                              -gammaln(n1alpha)-gammaln(n1lambda)+gammaln(n1alpha+n1lambda));                
                fnpc(id)=exp(log(n2alpha)+(n2alpha-1).*log(n2alpha*.1/n2lambda*n2mu)-...
                          log(n2lambda)-log(n2mu)-(n2alpha+n2lambda).*log(1+n2alpha*.1/n2lambda*n2mu)...
                              -gammaln(n2alpha)-gammaln(n2lambda)+gammaln(n2alpha+n2lambda));

                else 
                npc(id)=npc1;
                npnc(id)=npc1;
                fnpnc(id)=exp(log(n1alpha)+(n1alpha-1).*log(n1alpha*npc1/n1lambda*n1mu)-...
                          log(n1lambda)-log(n1mu)-(n1alpha+n1lambda).*log(1+n1alpha*npc1/n1lambda*n1mu)...
                              -gammaln(n1alpha)-gammaln(n1lambda)+gammaln(n1alpha+n1lambda));                
                fnpc(id)=exp(log(n2alpha)+(n2alpha-1).*log(n2alpha*npc1/n2lambda*n2mu)-...
                          log(n2lambda)-log(n2mu)-(n2alpha+n2lambda).*log(1+n2alpha*npc1/n2lambda*n2mu)...
                              -gammaln(n2alpha)-gammaln(n2lambda)+gammaln(n2alpha+n2lambda));
                end     
            end

            J(ind)=-(P_0_taw*log(P_0_taw)+H0_2ndterm+P_1_taw*log(P_1_taw)+H1_2ndterm);
            J=real(J);
            taw_vales(ind)=taw;
            [min_J(ind),min_error_taw_ind]=min(J);
            min_error_taw=taw_vales(min_error_taw_ind);

            if taw==0
            axes(handles.axes7);
            plotyy(xnc, fnc, npnc,fnpnc)
      
            elseif min_J(ind)< min_J(ind-1)
            axes(handles.axes7);
            plotyy(xnc, fnc, npnc,fnpnc)

            axes(handles.axes13);
             plotyy(xc, fc, npc,fnpc)

%             
% figure(4);
%  clf
%  plotyy( xnc, fnc,npnc,fnpnc)
% 
% figure(5);
%  clf
%  plotyy( xc, fc,npc, fnpc)


            end



            elseif val_pdf(2)==1
                %% main body of KI for Gamma pdf
            iii=0;%numerator            
            for q=0:taw
                P_0_taw=P_0_taw+h(q+1);
                if q~=0
                iii=iii+1;
                xnc(iii,1)=q;
                fnc(iii,1)=h2(q+1);
                xc(iii,1)=q;
                fc(iii,1)=h3(q+1);%log(((n1l^n1l)*q^(n1l-1)*exp(-n1l*q/n1mu))/gamma(n1l)*(n1mu^n1l))
                H0_2ndterm=H0_2ndterm+(h(q+1)*(-gammaln(n1l)+n1l*(log(n1l)-log(n1mu))+(n1l-1)*log(q)-(n1l*q/n1mu)));
                
                end
                if q==0%because of that our quantized data is: .1 1 2 3 ... 255     
                iii=iii+1;
                xnc(iii,1)=0.1;
                fnc(iii,1)=h2(q+1);
                xc(iii,1)=0.1;
                fc(iii,1)=h3(q+1);
                H0_2ndterm=0;
                H0_2ndterm=H0_2ndterm+(h(q+1)*(-gammaln(n1l)+n1l*(log(n1l)-log(n1mu))+(n1l-1)*log(.1)-(n1l*.1/n1mu)));
                
                end
            end

            %up to plot pdfs
            npnc=zeros(maxts+1,1);
            fnpnc=zeros(maxts+1,1);
            npc=zeros((maxts+1),1);
            fnpc=zeros((maxts+1),1);
            id=0;
            for npnc1=0:taw
                id=id+1;
                if npnc1==0
                npnc(id)=.1;
                npc(id)=.1; 
                fnpnc(id)=exp(-gammaln(n1l)+n1l*(log(n1l)-log(n1mu))+(n1l-1)*log(.1)-(n1l*.1/n1mu));
                F_npnc(id)=gammainc(n1l,(.1*n1l/n1mu));%Cumulative distribution
                fnpc(id)=exp(-gammaln(n2l)+n2l*(log(n2l)-log(n2mu))+(n2l-1)*log(.1)-(n2l*.1/n2mu));
                F_npc(id)=gammainc(n2l,(.1*n2l/n2mu));
                else
                npnc(id)=npnc1;
                npc(id)=npnc1;                
                fnpnc(id)=exp(-gammaln(n1l)+n1l*(log(n1l)-log(n1mu))+(n1l-1)*log(npnc1)-(n1l*npnc1/n1mu));
                F_npnc(id)=gammainc(n1l,(id*n1l/n1mu));%Cumulative distribution
                fnpc(id)=exp(-gammaln(n2l)+n2l*(log(n2l)-log(n2mu))+(n2l-1)*log(npnc1)-(n2l*npnc1/n2mu)); 
                F_npc(id)=gammainc(n2l,(id*n2l/n2mu));
                end            
            end
            P_1_taw=0;
            H1_2ndterm=0;

            for w=(taw+1):maxts
                P_1_taw=P_1_taw+h(w+1);
                if w~=0

                iii=iii+1;    
                xc(iii,1)=w;
                fc(iii,1)=h3(w+1);
                xnc(iii,1)=w;
                fnc(iii,1)=h2(w+1);
                H1_2ndterm=H1_2ndterm+(h(w+1)*(-gammaln(n2l)+n2l*(log(n2l)-log(n2mu))+(n2l-1)*log(w)-(n2l*w/n2mu)));

                end
                if w==0
                iii=iii+1;    
                xc(iii,1)=0.1;
                fc(iii,1)=h3(w+1);
                xnc(iii,1)=0.1;
                fnc(iii,1)=h2(w+1);                                    
                H1_2ndterm=0;
                H1_2ndterm=H1_2ndterm+(h(w+1)*(-gammaln(n2l)+n2l*(log(n2l)-log(n2mu))+(n2l-1)*log(.1)-(n2l*.1/n2mu)));
                end
            end
            %be up to plot estimated pdfs

            for npc1=(taw+1):maxts
                id=id+1;
                if npc1==0
                npc(id)=.1;
                npnc(id)=.1;
                fnpnc(id)=exp(-gammaln(n1l)+n1l*(log(n1l)-log(n1mu))+(n1l-1)*log(.1)-(n1l*.1/n1mu));
                F_npnc(id)=gammainc(n1l,(.1*n1l/n1mu));%Cumulative distribution
                fnpc(id)=exp(-gammaln(n2l)+n2l*(log(n2l)-log(n2mu))+(n2l-1)*log(.1)-(n2l*.1/n2mu));
                F_npc(id)=gammainc(n2l,(.1*n2l/n2mu));
                else 
                npc(id)=npc1;
                npnc(id)=npc1;
                fnpnc(id)=exp(-gammaln(n1l)+n1l*(log(n1l)-log(n1mu))+(n1l-1)*log(npc1)-(n1l*npc1/n1mu));
                F_npnc(id)=gammainc(n1l,(id*n1l/n1mu));%Cumulative distribution
                fnpc(id)=exp(-gammaln(n2l)+n2l*(log(n2l)-log(n2mu))+(n2l-1)*log(npc1)-(n2l*npc1/n2mu));
                F_npc(id)=gammainc(n2l,(id*n2l/n2mu));%Cumulative distribution
                end     
            end

            J(ind)=-(P_0_taw*log(P_0_taw)+H0_2ndterm+P_1_taw*log(P_1_taw)+H1_2ndterm);
            J=real(J);
            taw_vales(ind)=taw;
            [min_J(ind),min_error_taw_ind]=min(J);
            min_error_taw=taw_vales(min_error_taw_ind);
% save('xchange.mat','xc')
% save('fchange.mat','fc')
% save('xnochange.mat','xnc')
% save('fnochange.mat','fnc')
% save('npchange.mat','npc')
% save('fnpchange.mat','fnpc')
%save('F_npchange.mat','F_npc')
% save('npnochange.mat','npnc')
% save('fnpnochange.mat','fnpnc')
%save('F_npnochange.mat','F_npnc')

            if taw==0
            axes(handles.axes7);
            plotyy(xnc, fnc, npnc,fnpnc)
      
            elseif min_J(ind)< min_J(ind-1)
            axes(handles.axes7);
            plotyy(xnc, fnc, npnc,fnpnc)

            axes(handles.axes13);
             plotyy(xc, fc, npc,fnpc)

            
% % figure(4);
% %  clf
% %  plotyy( xnc, fnc,npnc,fnpnc)
% % 
% % figure(5);
% %  clf
% %  plotyy( xc, fc,npc, fnpc)


            end


            elseif val_pdf(3)==1
                %% KI main body by log normal pdf
             iii=0;%numerator  
             for q=0:taw
                P_0_taw=P_0_taw+h(q+1);
                if q~=0
                iii=iii+1;
                xnc(iii,1)=q;
                fnc(iii,1)=h2(q+1);
                xc(iii,1)=q;
                fc(iii,1)=h3(q+1);

                H0_2ndterm=H0_2ndterm-h(q+1).*(log(q)+log(sqrt(n1kisi))+log(sqrt(2*pi))+((log(q)-n1phi)^2/(2*n1kisi)));
                %H0_2ndterm=H0_2ndterm+(h(4*q+1)*log(n1alpha*(((n1alpha*q)/((n1lambda)*n1mu))^(n1alpha-1))/(((((n1alpha*q)/((n1lambda)*n1mu))+1)^(n1alpha+n1lambda))*(n1lambda)*n1mu*beta(n1alpha,n1lambda))));
                end
                if q==0%because of that our quantized data is: .1 1 2 3 ... 255     
                iii=iii+1;
                xnc(iii,1)=.1;
                fnc(iii,1)=h2(q+1);
                xc(iii,1)=.1;
                fc(iii,1)=h3(q+1);
                H0_2ndterm=0;
                H0_2ndterm=H0_2ndterm-h(q+1).*(log(.1)+log(sqrt(n1kisi))+log(sqrt(2*pi))+((log(.1)-n1phi)^2/(2*n1kisi)));
                 
                
                %H0_2ndterm=H0_2ndterm+(h(4*q+1)*(log(n1alpha*(((n1alpha*0.1)/((n1lambda)*n1mu))^(n1alpha-1))/(((((n1alpha*0.1)/((n1lambda)*n1mu))+1)^(n1alpha+n1lambda))*(n1lambda)*n1mu*beta(n1alpha,n1lambda)))));
                end
            end

            %up to plot pds
            npnc=zeros(maxts+1,1);
            fnpnc=zeros(maxts+1,1);
            npc=zeros((maxts+1),1);
            fnpc=zeros((maxts+1),1);
            id=0;
            for npnc1=0:taw
                id=id+1;
                if npnc1==0
                npnc(id)=.1;
                npc(id)=.1; 
                fnpnc(id)=real(exp(-(log(.1)-n1phi)^2/(2*n1kisi))/(.1*sqrt(2*pi*n1kisi)));
                fnpc(id)=real(exp(-(log(.1)-n2phi)^2/(2*n2kisi))/(.1*sqrt(2*pi*n2kisi)));
                
                else
                npnc(id)=npnc1;
                npc(id)=npnc1;
                fnpnc(id)=real(exp(-(log(npnc1)-n1phi)^2/(2*n1kisi))/(npnc1*sqrt(2*pi*n1kisi)));
                fnpc(id)=real(exp(-(log(npnc1)-n2phi)^2/(2*n2kisi))/(npnc1*sqrt(2*pi*n2kisi)));
                end
            end


            P_1_taw=0;
            H1_2ndterm=0;

            for w=(taw+1):maxts
                P_1_taw=P_1_taw+h(w+1);
                if w~=0
                iii=iii+1;    
                xnc(iii,1)=w;
                fnc(iii,1)=h2(w+1);
                xc(iii,1)=w;
                fc(iii,1)=h3(w+1);

                H1_2ndterm=H1_2ndterm-h(w+1).*(log(w)+log(sqrt(n2kisi))+log(sqrt(2*pi))+((log(w)-n2phi)^2/(2*n2kisi)));
                end
                if w==0
                iii=iii+1;    
                xc(iii,1)=.1;
                fc(iii,1)=h3(w+1);
                xnc(iii,1)=.1;
                fnc(iii,1)=h2(w+1);                                    
                H1_2ndterm=0;
                H1_2ndterm=H1_2ndterm-h(w+1).*(log(.1)+log(sqrt(n2kisi))+log(sqrt(2*pi))+((log(.1)-n2phi)^2/(2*n2kisi)));                
                end
            end
            %be up to plot estimated pdfs

            for npc1=(taw+1):maxts
                if npc1==0
                id=id+1;
                npc(id)=.1;
                npnc(id)=.1;
                fnpc(id)=real(exp(-(log(.1)+log(sqrt(n2kisi))+log(sqrt(2*pi))+((log(.1)-n2phi)^2/(2*n2kisi)))));
                fnpnc(id)=real(exp(-(log(.1)+log(sqrt(n1kisi))+log(sqrt(2*pi))+((log(.1)-n1phi)^2/(2*n1kisi)))));
                else
                   id=id+1; 
                npc(id)=npc1;
                npnc(id)=npc1;
                fnpc(id)=real(exp(-(log(npc1)+log(sqrt(n2kisi))+log(sqrt(2*pi))+((log(npc1)-n2phi)^2/(2*n2kisi)))));
                fnpnc(id)=real(exp(-(log(npc1)+log(sqrt(n1kisi))+log(sqrt(2*pi))+((log(npc1)-n1phi)^2/(2*n1kisi)))));
                end     
            end
   
            J(ind)=-(P_0_taw*log(P_0_taw)+H0_2ndterm+P_1_taw*log(P_1_taw)+H1_2ndterm);
            J=real(J);
            taw_vales(ind)=taw;
            [min_J(ind),min_error_taw_ind]=min(J);
            min_error_taw=taw_vales(min_error_taw_ind);
            if taw==0
            axes(handles.axes7);
            plotyy(xnc, fnc, npnc,fnpnc)
      
            elseif min_J(ind)< min_J(ind-1)
            axes(handles.axes7);
            plotyy(xnc, fnc, npnc,fnpnc)

            axes(handles.axes13);
             plotyy(xc, fc, npc,fnpc)

            
% % figure(4);
% %  clf
% %  plotyy( xnc, fnc,npnc,fnpnc)
% % 
% % figure(5);
% %  clf
% %  plotyy( xc, fc,npc, fnpc) 
            end
            elseif val_pdf(4)==1
                %% KI main body by weibull ratio pdf
                

                
            iii=0;%numerator            
            for q=0:taw
                P_0_taw=P_0_taw+h(q+1);
                if q~=0
                iii=iii+1;
                xnc(iii,1)=q;
                fnc(iii,1)=h2(q+1);
                xc(iii,1)=q;
                fc(iii,1)=h3(q+1);
                H0_2ndterm=H0_2ndterm+h(q+1).*(log(n1eta)+n1eta*log(n1lambda)+(n1eta-1)*log(q)-2*log(n1lambda.^n1eta+q.^n1eta));
                end
                if q==0%because of that our quantized data is: .1 1 2 3 ... 255     
                iii=iii+1;
                xnc(iii,1)=0.1;
                fnc(iii,1)=h2(q+1);
                xc(iii,1)=0.1;
                fc(iii,1)=h3(q+1);
                H0_2ndterm=0;
                H0_2ndterm=H0_2ndterm+h(q+1).*(log(n1eta)+n1eta*log(n1lambda)+(n1eta-1)*log(.1)-2*log(n1lambda.^n1eta+.1.^n1eta));

                end
            end

            %up to plot pdfs
            npnc=zeros(maxts+1,1);
            fnpnc=zeros(maxts+1,1);
            npc=zeros((maxts+1),1);
            fnpc=zeros((maxts+1),1);
            id=0;
            for npnc1=0:taw
                id=id+1;
                if npnc1==0
                npnc(id)=.1;
                npc(id)=.1;                
                fnpnc(id)=real(exp(log(n1eta)+n1eta*log(n1lambda)+(n1eta-1)*log(.1)-2*log(n1lambda.^n1eta+.1.^n1eta)));
                fnpc(id)=real(exp(log(n2eta)+n2eta*log(n2lambda)+(n2eta-1)*log(.1)-2*log(n2lambda.^n2eta+.1.^n2eta)));
                else
                npnc(id)=npnc1;
                npc(id)=npnc1;                
                fnpnc(id)=real(exp(log(n1eta)+n1eta*log(n1lambda)+(n1eta-1)*log(npnc1)-2*log(n1lambda.^n1eta+npnc1.^n1eta)));
                fnpc(id)=real(exp(log(n2eta)+n2eta*log(n2lambda)+(n2eta-1)*log(npnc1)-2*log(n2lambda.^n2eta+npnc1.^n2eta)));
                end            
            end
            P_1_taw=0;
            H1_2ndterm=0;

            for w=(taw+1):maxts
                P_1_taw=P_1_taw+h(w+1);
                if w~=0

                iii=iii+1;    
                xc(iii,1)=w;
                fc(iii,1)=h3(w+1);
                xnc(iii,1)=w;
                fnc(iii,1)=h2(w+1);
                H1_2ndterm=H1_2ndterm+h(w+1).*(log(n2eta)+n2eta*log(n2lambda)+(n2eta-1)*log(w)-2*log(n2lambda.^n2eta+w.^n2eta));
                end
                if w==0
                iii=iii+1;    
                xc(iii,1)=0.1;
                fc(iii,1)=h3(w+1);
                xnc(iii,1)=0.1;
                fnc(iii,1)=h2(w+1);                                    
                H1_2ndterm=0;
                H1_2ndterm=H1_2ndterm+h(w+1).*(log(n2eta)+n2eta*log(n2lambda)+(n2eta-1)*log(.1)-2*log(n2lambda.^n2eta+.1.^n2eta));                
                end
            end
            %be up to plot estimated pdfs

            for npc1=(taw+1):maxts
                id=id+1;
                if npc1==0
                npc(id)=.1;
                npnc(id)=.1;
                fnpnc(id)=real(exp(log(n1eta)+n1eta*log(n1lambda)+(n1eta-1)*log(.1)-2*log(n1lambda.^n1eta+.1.^n1eta)));
                fnpc(id)=real(exp(log(n2eta)+n2eta*log(n2lambda)+(n2eta-1)*log(.1)-2*log(n2lambda.^n2eta+.1.^n2eta)));
                else 
                npc(id)=npc1;
                npnc(id)=npc1;
                fnpnc(id)=real(exp(log(n1eta)+n1eta*log(n1lambda)+(n1eta-1)*log(npc1)-2*log(n1lambda.^n1eta+npc1.^n1eta)));
                fnpc(id)=real(exp(log(n2eta)+n2eta*log(n2lambda)+(n2eta-1)*log(npc1)-2*log(n2lambda.^n2eta+npc1.^n2eta)));
                end     
            end

            J(ind)=-(P_0_taw*log(P_0_taw)+H0_2ndterm+P_1_taw*log(P_1_taw)+H1_2ndterm);
            J=real(J);
            taw_vales(ind)=taw;
            [min_J(ind),min_error_taw_ind]=min(J);
            min_error_taw=taw_vales(min_error_taw_ind);

            if taw==0
            axes(handles.axes7);
            plotyy(xnc, fnc, npnc,fnpnc)
      
            elseif min_J(ind)< min_J(ind-1)
            axes(handles.axes7);
            plotyy(xnc, fnc, npnc,fnpnc)

            axes(handles.axes13);
             plotyy(xc, fc, npc,fnpc)

% figure(4);
%  clf
%  plotyy( xnc, fnc,npnc,fnpnc)
% 
% figure(5);
%  clf
%  plotyy( xc, fc,npc, fnpc)
% 

            end
                
                
                
              
            elseif val_pdf(5)==1
                %% KI main body for Generalized Gamma case 
            iii=0;%numerator            
            for q=0:taw
                P_0_taw=P_0_taw+h(q+1);
                if q~=0
                iii=iii+1;
                xnc1(iii,1)=q;
                fnc1(iii,1)=h_test_nc(q+1);
                xc1(iii,1)=q;
                fc1(iii,1)=h_test_c(q+1);
                H0_2ndterm=H0_2ndterm+h(q+1).*(log(abs(n1v))-gammaln(n1k)+n1k*n1v*(log(q)-log(n1sig))-log(q)-(q/n1sig)^n1v);
                
                end
                if q==0%because of that our quantized data is: .1 1 2 3 ... 255     
                iii=iii+1;
                xnc1(iii,1)=0.1;
                fnc1(iii,1)=h_test_nc(q+1);
                xc1(iii,1)=0.1;
                fc1(iii,1)=h_test_c(q+1);
                H0_2ndterm=0;
                H0_2ndterm=H0_2ndterm+h(q+1).*(log(abs(n1v))-gammaln(n1k)+n1k*n1v*(log(.1)-log(n1sig))-log(.1)-(.1/n1sig)^n1v);
                
                end
            end

            %up to plot pdfs
            npnc_1=zeros(maxts+1,1);
            fnpnc_1=zeros(maxts+1,1);
            npc_1=zeros((maxts+1),1);
            fnpc_1=zeros((maxts+1),1);
            id=0;
            for npnc1=0:taw
                id=id+1;
                if npnc1==0
                npnc_1(id)=.1;
                npc_1(id)=.1;                
                fnpnc_1(id)=exp(log(abs(n1v))-gammaln(n1k)+n1k*n1v*(log(.1)-log(n1sig))-log(.1)-(.1/n1sig)^n1v);
                F_npnc_1(id)=gammainc(n1k,(.1/n1sig)^n1v);%cumulative dist.
                fnpc_1(id)=exp(log(abs(n2v))-gammaln(n2k)+n2k*n2v*(log(.1)-log(n2sig))-log(.1)-(.1/n2sig)^n2v);
                F_npc_1(id)=gammainc(n2k,(.1/n2sig)^n2v);%cumulative dist.                
                else
                npnc_1(id)=npnc1;
                npc_1(id)=npnc1;                
                fnpnc_1(id)=exp(log(abs(n1v))-gammaln(n1k)+n1k*n1v*(log(npnc1)-log(n1sig))-log(npnc1)-(npnc1/n1sig)^n1v);
                F_npnc_1(id)=gammainc(n1k,(id/n1sig)^n1v);%cumulative dist.
                fnpc_1(id)=exp(log(abs(n2v))-gammaln(n2k)+n2k*n2v*(log(npnc1)-log(n2sig))-log(npnc1)-(npnc1/n2sig)^n2v);
                F_npc_1(id)=gammainc(n2k,(id/n2sig)^n2v);%cumulative dist.
                end            
            end

            P_1_taw=0;
            H1_2ndterm=0;

            for w=(taw+1):maxts
                P_1_taw=P_1_taw+h(w+1);
                if w~=0

                iii=iii+1;    
                xc1(iii,1)=w;
                fc1(iii,1)=h_test_c(w+1);
                xnc1(iii,1)=w;
                fnc1(iii,1)=h_test_nc(w+1);

                H1_2ndterm=H1_2ndterm+h(w+1).*(log(abs(n2v))-gammaln(n2k)+n2k*n2v*(log(w)-log(n2sig))-log(w)-(w/n2sig)^n2v);
                end
                if w==0
                iii=iii+1;    
                xc1(iii,1)=0.1;
                fc1(iii,1)=h_test_c(w+1);
                xnc1(iii,1)=0.1;
                fnc1(iii,1)=h_test_nc(w+1);                                    
                H1_2ndterm=0;
                H1_2ndterm=H1_2ndterm+h(w+1).*(log(abs(n2v))-gammaln(n2k)+n2k*n2v*(log(.1)-log(n2sig))-log(.1)-(.1/n2sig)^n2v);                
                end
            end
            %be up to plot estimated pdfs

            for npc1=(taw+1):maxts
                id=id+1;
                if npc1==0
                npc_1(id)=.1;
                npnc_1(id)=.1;
                fnpnc_1(id)=exp(log(abs(n1v))-gammaln(n1k)+n1k*n1v*(log(.1)-log(n1sig))-log(.1)-(.1/n1sig)^n1v);
                F_npnc_1(id)=gammainc(n1k,(.1/n1sig)^n1v);%cumulative dist.
                fnpc_1(id)=exp(log(abs(n2v))-gammaln(n2k)+n2k*n2v*(log(.1)-log(n2sig))-log(.1)-(.1/n2sig)^n2v);
                F_npc_1(id)=gammainc(n2k,(.1/n2sig)^n2v);%cumulative dist.
                else 
                npc_1(id)=npc1;
                npnc_1(id)=npc1;
                fnpnc_1(id)=exp(log(abs(n1v))-gammaln(n1k)+n1k*n1v*(log(npc1)-log(n1sig))-log(npc1)-(npc1/n1sig)^n1v);
                F_npnc_1(id)=gammainc(n1k,(id/n1sig)^n1v);%cumulative dist.
                fnpc_1(id)=exp(log(abs(n2v))-gammaln(n2k)+n2k*n2v*(log(npc1)-log(n2sig))-log(npc1)-(npc1/n2sig)^n2v);
                F_npc_1(id)=gammainc(n2k,(id/n2sig)^n2v);%cumulative dist.
                end     
            end

            J(ind)=-(P_0_taw*log(P_0_taw)+H0_2ndterm+P_1_taw*log(P_1_taw)+H1_2ndterm);
            J=real(J);
            taw_vales(ind)=taw;
%             J(1)=100000000000000000000000000000;
%             J(2)=100000000000000000000000000000;
%             J(3)=100000000000000000000000000000;            
            [min_J(ind),min_error_taw_ind]=min(J);
            min_error_taw=taw_vales(min_error_taw_ind);
% if taw==5
%    save('xnochange.mat','npnc_1')%no change
%    save('fxnochange.mat','fnpnc_1')%PDf estimates, no change
%    save('xchange.mat','npc_1')% change 
%    save('fxchange.mat','fnpc_1')%PDF estimates, change
%    save('nochange.mat','xnc1')%no change(gray levels)
%    save('hnochange.mat','fnc1')%Histogram values, no change
%    save('change.mat','xc1')%change(gray levels)
%    save('hchange.mat','fc1')%Histogram values, change
% end
%               
%             
%             
            
            
%             
% C_D=127.50*ones(r_m,c_m);
% for x=1:r_m
%     for y=1:c_m
%         if (ts(x,y))>taw
%         C_D(x,y)=255;
%         end
%     end
% end            
%             rc=0;fp=0;fn=0;tp=0;n_nc=0;
% for i=1:r_m
%     for j=1:c_m
%     if T_M(i,j)==255
%         rc=rc+1;
%         if C_D(i,j)==255
%             tp=tp+1;
%         elseif C_D(i,j)==127.5
%             fn=fn+1;
%         end
%     elseif T_M(i,j)==127.5
%         n_nc=n_nc+1;
%         if C_D(i,j)==255
%             fp=fp+1;
%         end
%     end
%     end
% end
% oerror(ind)=((fn+fp)/(n_nc+rc))*100;
% Thre(ind)=ind;
%             
            
            
            
            
            if taw==0
            axes(handles.axes7);
            plotyy(xnc1, fnc1, npnc_1,fnpnc_1)

            elseif min_J(ind)< min_J(ind-1)
            axes(handles.axes7);
            plotyy(xnc1, fnc1, npnc_1,fnpnc_1)

            axes(handles.axes13);
             plotyy(xc1, fc1, npc_1,fnpc_1)


 
%  save('xchange1.mat','xc1')
%  save('fchange1.mat','fc1')
%  save('xnochange1.mat','xnc1')
%  save('fnochange1.mat','fnc1')
%  save('npchange1.mat','npc_1')
%  save('fnpchange1.mat','fnpc_1')
% %save('F_npchange1.mat','F_npc_1')
%  save('npnochange1.mat','npnc_1')
%   save('fnpnochange1.mat','fnpnc_1')
% %save('F_npnochange1.mat','F_npnc_1')


% %loading the G parameters for plotting in a same axes
% load('xchange.mat')
% load('fchange.mat')
% load('xnochange.mat')
% load('fnochange.mat')
% load('npchange.mat')
% load('fnpchange.mat')
% load('npnochange.mat')
% load('fnpnochange.mat')
%          
% figure(4);
%  clf
%  plotyy( xnc1, fnc1,npnc_1,fnpnc_1)
% % hold on
% % plotyy( xnc, fnc,npnc,fnpnc)
% % hold off
% figure(5);
%  clf
%  plotyy( xc1, fc1,npc_1, fnpc_1)
% % hold on
% %  plotyy(xc, fc, npc, fnpc)
% % hold off


            end
            end
            ind=ind+1;
end

%save('Jfunction.mat','J')
%save('OverallError.mat','oerror')
%save('giventhreshold.mat','Thre')

%min_error_taw
%change or no-change map 
C_D=127.50*ones(r_m,c_m);
for x=1:r_m
    for y=1:c_m
        if (ts(x,y))>(min_error_taw)
        C_D(x,y)=255;
        end
    end
end
%save('CD2.mat','C_D')
fC_D=uint8(C_D);
axes(handles.axes6)
axis off;
imshow(fC_D);
figure(2)
clf
imshow(fC_D)
%load('CD1.mat')
% 
% fC_D=127.50*ones(r_m,c_m);
% for x=1:r_m
%     for y=1:c_m
%         if C_D(x,y)==255
%         fC_D(x,y)=255;
%         end
%         if CD1(x,y)==255
%         fC_D(x,y)=255;
%         end
%     end
% end
% fC_D=uint8(fC_D);
% axes(handles.axes6)
% axis off;
% imshow(fC_D);
% figure(1)
% clf
% imshow(fC_D)
% save('J.mat','J')
% save('H0.mat','H0_2ndterm')
% save('H1.mat','H1_2ndterm')
% save('P0.mat','P_0_taw')
% save('P1.mat','P_1_taw')
% %min_J
% J
% min_error_taw_ind
% min_error_taw
% 


elseif val_method(2)==1
    
    
    
    
    
    
    
   
val_ts(1,1)=get(handles.radiobutton40,'value');
val_ts(2,1)=get(handles.radiobutton41,'value');
if val_ts(1)==1

elseif val_ts(2)==1
     
end
end
%numerical evaluation
rc=0;fp=0;fn=0;tp=0;n_nc=0;
for i=1:r_m
    for j=1:c_m
    if T_M(i,j)==255
        rc=rc+1;
        if C_D(i,j)==255
            tp=tp+1;
        elseif C_D(i,j)==127.5
            fn=fn+1;
        end
    elseif T_M(i,j)==127.5
        n_nc=n_nc+1;
        if C_D(i,j)==255
            fp=fp+1;
        end
    end
    end
end


daccuracy=(tp/rc)*100;
falarm=(fp/n_nc)*100;
overallerror=((fn+fp)/(n_nc+rc))*100;


% if val_type(2)==1%ground truth corresponding to real data
%     
% rc=0;fp=0;fn=0;tp=0;n_nc=0;n_c=0;
% for i=1:r_m
%     for j=1:c_m
%     if T_M(i,j)==255
%         n_c=n_c+1;
%         rc=rc+1;
%         if C_D(i,j)==255
%             tp=tp+1;
%         elseif C_D(i,j)==127.5
%             fn=fn+1;
%         end
%     elseif T_M(i,j)==127.5
%         n_nc=n_nc+1;
%         if C_D(i,j)==255
%             fp=fp+1;
%         end
%     end
%     end
% end
% daccuracy=(tp/rc)*100;
% falarm=(fp/n_nc)*100;
% overallerror=((fn+fp)/(n_c+n_nc))*100;
% end


LoGT=isempty(find(T_M(:,:)~=0));%lack of ground truth
if LoGT==1
set(handles.edit4,'string','NotAvailable');
set(handles.edit5,'string','NotAvailable');
set(handles.edit6,'string','NotAvailable');
else
set(handles.edit4,'string',daccuracy);
set(handles.edit5,'string',falarm);
set(handles.edit6,'string',overallerror);
end
toc

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global a b a_m b_m r r_m c c_m d t_s ts L TS TSS t_s1 t_ss1 srw T_M;
double(t_s);
format long
%t_s: real part of complex trace statistic values
%t_s1: As we know HL trace statistic transformed no-change values to values
%centered around polarimetric dimension and the change values to
%values much smaller or much larger than that.
%ts1:This matrix is taken into account regarding that log10(uint) can not be defined
%TS=quantized trace statistic image(A^-1*B):this image is achieved by converting t_s1 values into 8bit integers  
%TSS:quantized trace statistic image(B^-1*A):this image is achieved by converting t_s1 values into 8bit integers
%ts:quantied trace statistic image(mix):for A^1*-B>1, values of t_s and if not values of t_ss are taken into aaccount in this image.
t_s=zeros(r_m*c_m,1);t_ss=zeros(r_m*c_m,1);
t_s1=zeros(r_m*c_m,1);t_ss1=zeros(r_m*c_m,1);
t_S1=zeros(r_m*c_m,1);t_SS1=zeros(r_m*c_m,1);
ts=zeros(r_m,c_m);
mix_ts=zeros(r_m,c_m);
srw=zeros(r_m*c_m,1);
bart=zeros(r_m*c_m,1);
val_case=zeros(6,1);
val_case(1,1)=get(handles.radiobutton73,'value');
val_case(2,1)=get(handles.radiobutton72,'value');
val_case(3,1)=get(handles.radiobutton69,'value');
val_case(4,1)=get(handles.radiobutton74,'value');
val_case(5,1)=get(handles.radiobutton70,'value');
val_case(6,1)=get(handles.radiobutton71,'value');





if d==1
    
    
    
aa_m=reshape(a_m,r_m*c_m,1);
bb_m=reshape(b_m,r_m*c_m,1);
a_m2=zeros(r_m,c_m);
b_m2=zeros(r_m,c_m);
for i=1:r_m*c_m
        a_m2(i)=aa_m{i};
        b_m2(i)=bb_m{i};
        if a_m2(i)~=0 && b_m2(i)~=0
        t_s(i)=b_m2(i)/a_m2(i);
        t_ss(i)=a_m2(i)/b_m2(i);
        t_s1(i)=t_s(i)-d;
        t_ss1(i)=t_ss(i)-d;
        srw(i)=((b_m2(i)/a_m2(i))+(a_m2(i)/b_m2(i))/2)-d;
        end
end






else
 load('TwoCoregisteredMultilookedCovarianceData.mat')
% s=size(C1);
% r_m=s(3);
% c_m=s(4);

if val_case (1,1)==1
clear a_m b_m
%d=1;
a_m=cell(r_m,c_m);
b_m=cell(r_m,c_m);
for i=1:r_m
    for j=1:c_m
        a_m{i,j}=C1(1,1,i,j);
        b_m{i,j}=C2(1,1,i,j);
    end
end   

elseif val_case (2,1)==1
clear a_m b_m
%d=1;
a_m=cell(r_m,c_m);
b_m=cell(r_m,c_m);
for i=1:r_m
    for j=1:c_m
        a_m{i,j}=C1(2,2,i,j);
        b_m{i,j}=C2(2,2,i,j);
    end
end   
    
elseif val_case (3,1)==1
clear a_m b_m
%d=1;
a_m=cell(r_m,c_m);
b_m=cell(r_m,c_m);
for i=1:r_m
    for j=1:c_m
        a_m{i,j}=C1(3,3,i,j);
        b_m{i,j}=C2(3,3,i,j);
    end
end   

elseif val_case (4,1)==1
clear a_m b_m
a_m=cell(r_m,c_m);
b_m=cell(r_m,c_m);
for i=1:r_m
    for j=1:c_m
        a_m{i,j}=C1(:,:,i,j);
        b_m{i,j}=C2(:,:,i,j);
        a_m{i,j}(1,2)=0;a_m{i,j}(1,3)=0;a_m{i,j}(2,3)=0;
        a_m{i,j}(2,1)=0;a_m{i,j}(1,3)=0;a_m{i,j}(3,2)=0;
        b_m{i,j}(1,2)=0;b_m{i,j}(1,3)=0;b_m{i,j}(2,3)=0;
        b_m{i,j}(2,1)=0;b_m{i,j}(1,3)=0;b_m{i,j}(3,2)=0;
    end
end

elseif val_case (5,1)==1%azimuthal case
clear a_m b_m
a_m=cell(r_m,c_m);
b_m=cell(r_m,c_m);
for i=1:r_m
    for j=1:c_m
        a_m{i,j}(1,1)=C1(1,1,i,j);a_m{i,j}(1,2)=C1(1,3,i,j);a_m{i,j}(1,3)=0;
        a_m{i,j}(2,1)=C1(3,1,i,j);a_m{i,j}(2,2)=C1(3,3,i,j);a_m{i,j}(2,3)=0;
        a_m{i,j}(3,1)=0;a_m{i,j}(3,2)=0;a_m{i,j}(3,3)=C1(2,2,i,j);
        
        b_m{i,j}(1,1)=C2(1,1,i,j);b_m{i,j}(1,2)=C2(1,3,i,j);b_m{i,j}(1,3)=0;
        b_m{i,j}(2,1)=C2(3,1,i,j);b_m{i,j}(2,2)=C2(3,3,i,j);b_m{i,j}(2,3)=0;
        b_m{i,j}(3,1)=0;b_m{i,j}(3,2)=0;b_m{i,j}(3,3)=C2(2,2,i,j);
    end
end

end
aa_m=reshape(a_m,r_m*c_m,1);
bb_m=reshape(b_m,r_m*c_m,1);
% (det(aa_m{1}+bb_m{1}))^2
% det(aa_m{1})*det(bb_m{1})
% log(((det(aa_m{1}+bb_m{1}))^2)/(det(aa_m{1})*det(bb_m{1})))
% 
% (det(aa_m{2}+bb_m{2}))^2
% det(aa_m{2})*det(bb_m{2})
% log(((det(aa_m{2}+bb_m{2}))^2)/(det(aa_m{2})*det(bb_m{2})))
% 
% (det(aa_m{3}+bb_m{3}))^2
% det(aa_m{3})*det(bb_m{3})
% log(((det(aa_m{3}+bb_m{3}))^2)/(det(aa_m{3})*det(bb_m{3})))
for i=1:r_m*c_m       
    t_s(i)=real(trace(aa_m{i}\bb_m{i}));
    t_ss(i)=real(trace(bb_m{i}\aa_m{i}));
    t_S1(i)=t_s(i)-d;
    t_SS1(i)=t_ss(i)-d;
    srw(i)=((real(trace(aa_m{i}\bb_m{i}))+real(trace(bb_m{i}\aa_m{i})))/2)-d;
    bart(i)=real(log(((det(aa_m{i}+bb_m{i}))^2)/(det(aa_m{i})*det(bb_m{i}))));
end
end
t1=reshape(t_s,r_m,c_m);%for showing in the figure
t2=reshape(t_ss,r_m,c_m);%for showing in the figure
m1_ts=zeros(r_m,c_m);%for showing in the figure
for i=1:r_m
        for j=1:c_m
            if t1(i,j)>=t2(i,j)
            m1_ts(i,j)=t1(i,j);%mixed ts
            else
            m1_ts(i,j)=t2(i,j);
            end
        end
end

t_s1=reshape(t_S1,r_m,c_m);
t_ss1=reshape(t_SS1,r_m,c_m);
srw=reshape(srw,r_m,c_m);
bart=reshape(bart,r_m,c_m);
%save('bart.mat','bart')
% figure(1)
% bar(t_s)
% figure(2)
% bar(t_ss)
% figure(3)
% bar(srw)
% save('ts1.mat','t_s1')

% dlmwrite('t_s.txt', t_ss, 'delimiter', '\t');
value=zeros(3,1);
value(1,1)=get(handles.radiobutton40,'value');
value(2,1)=get(handles.radiobutton41,'value');
value(3,1)=get(handles.radiobutton75,'value');
if value(1)==1
    for i=1:r_m
        for j=1:c_m
            if t_s1(i,j)>=t_ss1(i,j)
            mix_ts(i,j)=t_s1(i,j);%mixed ts
            else
            mix_ts(i,j)=t_ss1(i,j);
            end
        end
    end

TS=zeros(r_m,c_m);TSS=zeros(r_m,c_m);

maxts1=max(max(t_s1));
maxtss1=max(max(t_ss1));
maxts_s=max(maxts1,maxtss1);
maxts_s=fix(maxts_s)+1;
ts1=zeros(r_m,c_m);
tss1=zeros(r_m,c_m);
m_ts=zeros(r_m,c_m);
for i=1:r_m
    for j=1:c_m
        for iter=0:maxts_s
            if t_s1(i,j)<(iter+1) && t_s1(i,j)>=iter
                ts1(i,j)=iter;
            end
            if t_ss1(i,j)<(iter+1) && t_ss1(i,j)>=iter
                tss1(i,j)=iter;
            end
            if mix_ts(i,j)<(iter+1) && mix_ts(i,j)>=iter
                m_ts(i,j)=iter;
            end            
        end
    end
end

for i=1:r_m
    for j=1:c_m
        TS(i,j)=ts1(i,j);
        TSS(i,j)=tss1(i,j);
        ts(i,j)=m_ts(i,j);
        if TS(i,j)==0
            TS(i,j)=0.1;
        end
        if TSS(i,j)==0
            TSS(i,j)=0.1;
        end
        if ts(i,j)==0
            ts(i,j)=0.1;
        end
    end
end


% HLTts=10*log10(double(m1_ts));
% save('HLTts.mat','HLTts')

%dlmwrite('HL.txt', ts, 'delimiter', '\t');
%showing trace statistic image
% %ts=TS;
% t1(1:10,1:10)
% [rr,cc]=find(t1<0)
% t1(rr,cc)=0.001;
% [rr1,cc1]=find(t2<0);
% t2(rr1,cc1)=0.001;
% t2(1:10,1:10)
% % 
% % for i=r_m
% %     for j=1:c_m
% %         if t1(i,j)>0
% %         else
% %             t1(i,j)
% %             t1(i,j)=0.001;
% %         end
% %         
% %         if t2(i,j)>0
% %         else
% %             t2(i,j)=0.001;
% %         end
% %     end
% % end
% % t1(1:10,1:10)
axes(handles.axes3);
imagesc(real(10*log10(double(real(t1)))));%doubleTS
axis off
axis equal
axes(handles.axes4);
imagesc(real(10*log10(double(real(t2)))));%TSS
axis off
axis equal
axes(handles.axes8);
imagesc(real(10*log10(double(m1_ts))));%ts
axis off
axis equal
% figure(10)
% imagesc(log(double(t1)));
% axis off
% axis equal
% figure(11)
% imagesc(log(double(t2)));
% axis off
% axis equal
figure(12)
imagesc(real(10*log10(double(real(t1)))));axis off
axis equal
figure(13)
imagesc(real(10*log10(double(real(t2)))));
axis off
axis equal
figure(14)
imagesc(real(10*log10(double(m1_ts))));
axis off
axis equal
elseif value(2)==1

maxsrw=max(max(srw));
maxsrw=fix(maxsrw)+1;
ts1=zeros(r_m,c_m);
for i=1:r_m
    for j=1:c_m
        for iter=0:maxsrw
            if srw(i,j)<(iter+1) && srw(i,j)>=iter
                ts1(i,j)=iter;
            end
        end
    end
end



for i=1:r_m
    for j=1:c_m
        ts(i,j)=ts1(i,j);
        if ts1(i,j)==0
           ts(i,j)=.1;
        end
    end
end
% dlmwrite('SRW1.txt', ts, 'delimiter', '\t');

%showing trace statistic image

% 
% 
% 
% 
% 
% %histogram of quantized trace image
% %ts=TS;
% maxts=max(max(srw));
% %max(max(ts));
% h1=zeros(100*maxts+1,1);
% kk=zeros(100*maxts+1,1);
% i=0;
% for k=0:.01:maxts
%     i=i+1;
%     kk(i)=k;
%     for l=1:r_m
%         for m=1:c_m
% %             if k==0
% %                 kk(k+1)=.1;
% %                 if (srw(l,m))==.1 && T_M(l,m)~=0
% %                     h1(k+1)=h1(k+1)+1;
% %                 end
% %             end
%             if (srw(l,m))>k && srw(l,m)<k+.01
%                 if T_M(l,m)~=0
%                 h1(i)=h1(i)+1;
%                 end
%             end
%         end
%     end
% end
% %h=h1/(r_m*c_m);
% h=log10(h1/(size(find(T_M~=0),1)));
% % axes(handles.axes5);
% %  k_pr=10*log10(kk);
% % % h_pr=10*log10(h);
% % bar(kk,h1,'b');
%  
%  figure(10);
%  clf
%  linenu = 1.5;fs=20;
%  %[fA,xA] = [h,kk];
% % [fB,xB] = ksdensity(Mj2,'npoints',10000);
% % [fC,xC] = ksdensity(Mj5,'npoints',10000);
%  xvector=log10(kk');
%  fvector=h1';
%  plot(xvector, fvector, 'LineWidth',linenu);
%  %legend({'Histogram of TS image'},'FontSize',fs)
%  xlabel(['Gray level'], 'fontsize',fs)
%  ylabel('Frequency', 'fontsize',fs)
% set(gca,'FontSize',fs);
% 
% 
% 
% 





figure(1)
clf
imagesc(10*log10(double(abs(srw))))
axis off
axis equal
axes(handles.axes8);
imagesc(10*log10(double(abs(ts))));
axis off
axis equal
elseif value(3)==1
maxbart=max(max(bart));
maxbart=fix(maxbart)+1;
ts1=zeros(r_m,c_m);
for i=1:r_m
    for j=1:c_m
        for iter=0:maxbart
            if bart(i,j)<(iter+1) && bart(i,j)>=iter
                ts1(i,j)=iter;
            end
        end
    end
end



for i=1:r_m
    for j=1:c_m
        ts(i,j)=ts1(i,j);
        if ts1(i,j)==0
           ts(i,j)=.1;
        end
    end
end
% dlmwrite('SRW1.txt', ts, 'delimiter', '\t');

%showing trace statistic image
figure(1)
clf
imagesc(10*log10(double(ts)))
axis off
axes(handles.axes8);
imagesc(1000*log10(double(ts)));
axis off  
end


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global r c r_m c_m a_m b_m a b d L L1 oac_check T_M ;% r and c specify number of rows and columns respectively and L,L1, d, a and b  specify number of looks in Azi and range direction and size of cov matrix and cov matrix in first and 2nd images respectively. 

L=str2double(get(handles.edit1,'String'));
L1=str2double(get(handles.edit2,'String'));
oac_check=get(handles.checkbox1,'value');

value=zeros(6,1);
value(1,1)=get(handles.radiobutton1,'value');
value(2,1)=get(handles.radiobutton2,'value');
value(3,1)=get(handles.radiobutton3,'value');
value(4,1)=get(handles.radiobutton4,'value');
value(5,1)=get(handles.radiobutton5,'value');
value(6,1)=get(handles.radiobutton6,'value');

if value(1)==1
    d=1;
[FileName path]=uigetfile('*.bin','Import Intensity file(time1)');
a1=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
s=size(a1);
r=s(1);
c=s(2);

%simulating of 2nd cov data (b2) by copying a part of first data and pasting that in another part and vice versa. 
b2=a1;
for i=1:50
    for j=1:50
    b2(i,j,:)=a1((i+r-50),(j+c-50),:);
    b2(i+r-50,j+c-50,:)=a1(i,j,:);
    end
end

a=cell(r,c);
b=cell(r,c);
for i=1:r
    for j=1:c
        a{i,j}=a1(i,j);
        b{i,j}=b2(i,j);
    end
end
r_m=r;
c_m=c;
a_m=a;
b_m=b;    
elseif value(2)==1
    d=1;
[FileName path]=uigetfile('*.bin','Import Intensity file(time1)');
a1=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
s=size(a1);
r=s(1);
c=s(2);

%simulating of 2nd cov data (b2) by copying a part of first data and pasting that in another part and vice versa. 
b2=zeros(r,c,1);
b2=a1;
for i=1:50
    for j=1:50
    b2(i,j,:)=a1((i+r-50),(j+c-50),:);
    b2(i+r-50,j+c-50,:)=a1(i,j,:);
    end
end

a=cell(r,c);
b=cell(r,c);
for i=1:r
    for j=1:c
        a{i,j}=a1(i,j);
        b{i,j}=b2(i,j);
    end
end
r_m=r;
c_m=c;
a_m=a;
b_m=b;        
elseif value(3)==1

     d=1;   
[FileName path]=uigetfile('*.bin','Import Intensity file(time1)');
a1=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
s=size(a1);
r=s(1);
c=s(2);

%simulating of 2nd cov data (b2) by copying a part of first data and pasting that in another part and vice versa. 
b2=zeros(r,c,1);
b2=a1;
for i=1:50
    for j=1:50
    b2(i,j,:)=a1((i+r-50),(j+c-50),:);
    b2(i+r-50,j+c-50,:)=a1(i,j,:);
    end
end

a=cell(r,c);
b=cell(r,c);
for i=1:r
    for j=1:c
        a{i,j}=a1(i,j);
        b{i,j}=b2(i,j);
    end
end
 r_m=r;
c_m=c;
a_m=a;
b_m=b;       
elseif value(4)==1
    d=2;
[FileName path]=uigetfile('*.bin','Import C11.bin file');
C11=multibandread([path FileName],[904,523,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata

[FileName path]=uigetfile('*.bin','Import C12_real.bin file');
C12_real=multibandread([path FileName],[904,523,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
[FileName path]=uigetfile('*.bin','Import C12_imag.bin file');
C12_imag=multibandread([path FileName],[904,523,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata

[FileName path]=uigetfile('*.bin','Import C22.bin file');
C22=multibandread([path FileName],[904,523,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata

s=size(C11);
r=s(1);
c=s(2);

a2=zeros(r,c,4);
a2(:,:,1)=C11(:,:,1);%c11
a2(:,:,2)=C12_real(:,:,1)+1i*C12_imag(:,:,1);%c12
a2(:,:,3)=conj(a2(:,:,2));%c21
a2(:,:,4)=C22(:,:,1);%c22

%simulating of 2nd cov data (b2) by copying a part of first data and pasting that in another part and vice versa. 
b2=zeros(r,c,4);
b2=a2;
for i=1:50
    for j=1:50
    b2(i,j,:)=a2((i+r-50),(j+c-50),:);
    b2(i+r-50,j+c-50,:)=a2(i,j,:);
    end
end

a=cell(r,c);
b=cell(r,c);
for i=1:r
    for j=1:c
        a{i,j}=reshape(a2(i,j,:),2,2);
        b{i,j}=reshape(b2(i,j,:),2,2);
    end
end
r_m=r;
c_m=c;
a_m=a;
b_m=b;        
elseif value(5)==1
    d=2;
[FileName path]=uigetfile('*.bin','Import C22_2.bin file');
C22=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata

[FileName path]=uigetfile('*.bin','Import C23_real_2.bin file');
C23_real=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
[FileName path]=uigetfile('*.bin','Import C23_imag_2.bin file');
C23_imag=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata

[FileName path]=uigetfile('*.bin','Import C33_2.bin file');
C33=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata

s=size(C22);
r=s(1);
c=s(2);

a2=zeros(r,c,4);
a2(:,:,1)=C22(:,:,1);%c11
a2(:,:,2)=C23_real(:,:,1)+1i*C23_imag(:,:,1);%c23
a2(:,:,3)=conj(a2(:,:,2));%c32
a2(:,:,4)=C33(:,:,1);%c33

%simulating of 2nd cov data (b2) by copying a part of first data and pasting that in another part and vice versa. 
b2=zeros(r,c,4);
b2=a2;
for i=1:50
    for j=1:50
    b2(i,j,:)=a2((i+r-50),(j+c-50),:);
    b2(i+r-50,j+c-50,:)=a2(i,j,:);
    end
end

a=cell(r,c);
b=cell(r,c);
for i=1:r
    for j=1:c
        a{i,j}=reshape(a2(i,j,:),2,2);
        b{i,j}=reshape(b2(i,j,:),2,2);
    end
end
r_m=r;
c_m=c;
a_m=a;
b_m=b;

elseif value(6)==1
    d=3;
% [FileName path]=uigetfile('*.bin','Import C11.bin file');
% C11=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
C11=multibandread('C11_2.bin',[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});
% [FileName path]=uigetfile('*.bin','Import C12_real.bin file');
% C12_real=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
% [FileName path]=uigetfile('*.bin','Import C12_imag.bin file');
% C12_imag=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
C12_real=multibandread('C12_real_2.bin',[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});
C12_imag=multibandread('C12_imag_2.bin',[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});
% [FileName path]=uigetfile('*.bin','Import C13_real.bin file');
% C13_real=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
% [FileName path]=uigetfile('*.bin','Import C13_imag.bin file');
% C13_imag=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
C13_real=multibandread('C13_real_2.bin',[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});
C13_imag=multibandread('C13_imag_2.bin',[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});
% [FileName path]=uigetfile('*.bin','Import C22.bin file');
% C22=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
C22=multibandread('C22_2.bin',[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});
% [FileName path]=uigetfile('*.bin','Import C23_real.bin file');
% C23_real=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
% [FileName path]=uigetfile('*.bin','Import C23_imag.bin file');
% C23_imag=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
C23_real=multibandread('C23_real_2.bin',[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});
C23_imag=multibandread('C23_imag_2.bin',[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});
% [FileName path]=uigetfile('*.bin','Import C33.bin file');
% C33=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
C33=multibandread('C33_2.bin',[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});

s=size(C11);
r=s(1);
c=s(2);

a2=zeros(r,c,9);
a2(:,:,1)=C11(:,:,1);%c11
a2(:,:,2)=C12_real(:,:,1)+1i*C12_imag(:,:,1);%c12
a2(:,:,3)=C13_real(:,:,1)+1i*C13_imag(:,:,1);%c13
a2(:,:,4)=conj(a2(:,:,2));%c21
a2(:,:,5)=C22(:,:,1);%c22
a2(:,:,6)=C23_real(:,:,1)+1i*C23_imag(:,:,1);%c23
a2(:,:,7)=conj(a2(:,:,3));%c31
a2(:,:,8)=conj(a2(:,:,6));%c32
a2(:,:,9)=C33(:,:,1);%c33

%simulating of 2nd cov data (b2) by copying a part of first data and pasting that in another part and vice versa. 
b2=zeros(r,c,9);
b2=a2;
for i=1:50
    for j=1:50
    b2(i,j,:)=a2((i+r-50),(j+c-50),:);
    b2(i+r-50,j+c-50,:)=a2(i,j,:);
    end
end

a=cell(r,c);
b=cell(r,c);
for i=1:r
    for j=1:c
        a{i,j}=reshape(a2(i,j,:),3,3);
        b{i,j}=reshape(b2(i,j,:),3,3);
    end
end
    
end

% a5050=a{50,50}
% b201151=b{201,151}

%multilooking:in this section, we're gonna accomplish multilooking stage in
%the processing
%first:multilooking in azimuth direction
if (r/L)-fix(r/L)~=0
    
    r_m=1+fix(r/L);%r_m:number of rows in multilooked case
    
    a_m1=cell(r_m,c);%a_m:1st image in multilooked case
    b_m1=cell(r_m,c);%a_m:2nd image in multilooked case
 
    for j=1:c
        sum=0;
        for n=1:L
            sum=sum+a{n,j};
        end
        a_m1{1,j}=sum/L;
    end
    for i=2:r_m
        for j=1:c
           sum=0;
           if i==r_m
           for n=((r-L)+1):r
           sum=sum+a{n,j};
           end
           a_m1{i,j}=sum/L;
           continue
           end
           ii=((i-1)*L)+1;%from the row with this amount by L rows shold be averaged.(example:L=6, fori=2 n shold vary for ((2-1)*6+1=)7 by (7+6-1=)12)
           for n=ii:(ii+L-1)
           sum=sum+a{n,j};
           end
           a_m1{i,j}=sum/L;
        end 
    end
     
    

b_m1=a_m1;
for i=1:10
    for j=1:10
    b_m1{i,j}=a_m1{(i+r_m-10),(j+c-10)};
    b_m1{i+r_m-10,j+c-10}=a_m1{i,j};
    end
end

    
else
    r_m=r/L;%r_m:number of rows in multilooked case
    
    a_m1=cell(r_m,c);%a_m:1st image in multilooked case
    b_m1=cell(r_m,c);%a_m:2nd image in multilooked case
    if r_m==r
        a_m1=a;
        b_m1=b;
    else
    for j=1:c
        sum=0;
        for n=1:L
            sum=sum+a{n,j};
        end
        a_m1{1,j}=sum/L;
    end
    for i=2:r_m
        for j=1:c
           sum=0;
           ii=((i-1)*L)+1;%from the row with this amount by L rows should be averaged.(example:L=6, fori=2 n should vary for ((2-1)*6+1=)7 by (7+6-1=)12)
           for n=ii:(ii+L-1)
           sum=sum+a{n,j};
           end
           a_m1{i,j}=sum/L;
        end 
    end

    
b_m1=a_m1;
for i=1:10
    for j=1:10
    b_m1{i,j}=a_m1{(i+r_m-10),(j+c-10)};
    b_m1{i+r_m-10,j+c-10}=a_m1{i,j};
    end
end
    end
end


%second:multilooking in range direction
if (c/L1)-fix(c/L1)~=0
    
    c_m=1+fix(c/L1);%r_m:number of rows in multilooked case
    
    a_m=cell(r_m,c_m);%a_m:1st image in multilooked case
    b_m=cell(r_m,c_m);%a_m:2nd image in multilooked case
 
    for j=1:r_m
        sum=0;
        for n=1:L1
            sum=sum+a_m1{j,n};
        end
        a_m{j,1}=sum/L1;
    end
    for i=1:r_m
        for j=2:c_m
           sum=0;
           if j==c_m
           for n=((c-L1)+1):c
           sum=sum+a_m1{i,n};
           end
           a_m{i,j}=sum/L1;
           continue
           end
           ii=((j-1)*L1)+1;%from the row with this amount by L rows shold be averaged.(example:L=6, fori=2 n shold vary for ((2-1)*6+1=)7 by (7+6-1=)12)
           for n=ii:(ii+L1-1)
           sum=sum+a_m1{i,n};
           end
           a_m{i,j}=sum/L1;
        end 
    end
     
    

b_m=a_m;
for i=1:10
    for j=1:10
    b_m{i,j}=a_m{(i+r_m-10),(j+c_m-10)};
    b_m{i+r_m-10,j+c_m-10}=a_m{i,j};
    end
end

    
else
    c_m=c/L1;%r_m:number of rows in multilooked case

    a_m=cell(r_m,c_m);%a_m:1st image in multilooked case
    b_m=cell(r_m,c_m);%a_m:2nd image in multilooked case
    if c_m==c
        a_m=a_m1;
        b_m=b_m1;
    else
    for j=1:c_m
        sum=0;
        for n=1:L1
            sum=sum+a_m1{j,n};
        end
        a_m{j,1}=sum/L1;
    end
    for i=1:r_m
        for j=2:c_m
           sum=0;
           ii=((j-1)*L1)+1;%from the row with this amount by L rows should be averaged.(example:L=6, fori=2 n should vary for ((2-1)*6+1=)7 by (7+6-1=)12)
           for n=ii:(ii+L1-1)
           sum=sum+a_m1{i,n};
           end
           a_m{i,j}=sum/L1;
        end 
    end

    
b_m=a_m;
for i=1:10
    for j=1:10
    b_m{i,j}=a_m{(i+r_m-10),(j+c_m-10)};
    b_m{i+r_m-10,j+c_m-10}=a_m{i,j};
    end
end
    end
end

 size(a_m)
  size(b_m)
 
 
 a88_1=a_m{8,8}
 b9974_1=b_m{99,74}
%cmp=strcmp(oac_check,'Orientation angle compensation')
if oac_check==1

% [FileName path]=uigetfile('*.bin','Import T23_real.bin file');
% T23_real=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});
T23_real=multibandread('T23_real_2.bin',[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});
% [FileName path]=uigetfile('*.bin','Import T22.bin file');
% T22=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});
T22=multibandread('T22_2.bin',[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});
% [FileName path]=uigetfile('*.bin','Import T33.bin file');
% T33=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});
T33=multibandread('T33_2.bin',[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});
%multilooking of T matrix data:in this section, we're gonna accomplish multilooking of T23_real, T22 and T33 in
%the processing of OAC.
if (r/L)-fix(r/L)~=0
    
    r_m=1+fix(r/L);%r_m:number of rows in multilooked case
    
    T23_m1=zeros(r_m,c);%T23_m:multilooked T23_real data
    T22_m1=zeros(r_m,c);%T22_m:multilooked T22 data
    T33_m1=zeros(r_m,c);%T23_m:multilooked T33 data
    
    for j=1:c
        sum1=0;
        sum2=0;
        sum3=0;
        for n=1:L
            sum1=sum1+T23_real(n,j);
            sum2=sum2+T22(n,j);
            sum3=sum3+T33(n,j);
        end
        T23_m1(1,j)=sum1/L;
        T22_m1(1,j)=sum2/L;
        T33_m1(1,j)=sum3/L;
        
    end
    for i=2:r_m
        for j=1:c
           sum1=0;
           sum2=0;
           sum3=0;
           
           if i==r_m
           for n=((r-L)+1):r
           sum1=sum1+T23_real(n,j);
           sum2=sum2+T22(n,j);
           sum3=sum3+T33(n,j);
           end
           T23_m1(i,j)=sum1/L;
           T22_m1(i,j)=sum2/L;
           T33_m1(i,j)=sum3/L;
           continue
           end
           
           ii=((i-1)*L)+1;%from the row with this amount by L rows should be averaged.(example:L=6, fori=2 n should vary for ((2-1)*6+1=)7 by (7+6-1=)12)
           for n=ii:(ii+L-1)
           sum1=sum1+T23_real(n,j);
           sum2=sum2+T22(n,j);
           sum3=sum3+T33(n,j);
           end
           T23_m1(i,j)=sum1/L;
           T22_m1(i,j)=sum2/L;
           T33_m1(i,j)=sum3/L;
        end 
    end
     
    
else
    r_m=r/L;%r_m:number of rows in multilooked case
    
    T23_m1=zeros(r_m,c);%T23_m:multilooked T23_real data
    T22_m1=zeros(r_m,c);%T22_m:multilooked T22 data
    T33_m1=zeros(r_m,c);%T23_m:multilooked T33 data
     
    for j=1:c
        sum1=0;
        sum2=0;
        sum3=0;
        for n=1:L
            sum1=sum1+T23_real(n,j);
            sum2=sum2+T22(n,j);
            sum3=sum3+T33(n,j);
        end
        T23_m1(1,j)=sum1/L;
        T22_m1(1,j)=sum2/L;
        T33_m1(1,j)=sum3/L;
    end
    for i=2:r_m
        for j=1:c
        sum1=0;
        sum2=0;
        sum3=0;
        ii=((i-1)*L)+1;%from the row with this amount by L rows should be averaged.(example:L=6, fori=2 n should vary for ((2-1)*6+1=)7 by (7+6-1=)12)
        for n=ii:(ii+L-1)
           sum1=sum1+T23_real(n,j);
           sum2=sum2+T22(n,j);
           sum3=sum3+T33(n,j);
        end
           T23_m1(i,j)=sum1/L;
           T22_m1(i,j)=sum2/L;
           T33_m1(i,j)=sum3/L;
        end 
    end

end

%range directon multilooking of T matrix elements.
if (c/L1)-fix(c/L1)~=0
    
    c_m=1+fix(c/L1);%c_m:number of coloumns in multilooked case
    
    T23_m=zeros(r_m,c_m);%T23_m:multilooked T23_real data
    T22_m=zeros(r_m,c_m);%T22_m:multilooked T22 data
    T33_m=zeros(r_m,c_m);%T23_m:multilooked T33 data
    
    for j=1:r_m
        sum1=0;
        sum2=0;
        sum3=0;
        for n=1:L1
            sum1=sum1+T23_m1(j,n);
            sum2=sum2+T22_m1(j,n);
            sum3=sum3+T33_m1(j,n);
        end
        T23_m(j,1)=sum1/L1;
        T22_m(j,1)=sum2/L1;
        T33_m(j,1)=sum3/L1;
        
    end
    for i=1:r_m
        for j=2:c_m
           sum1=0;
           sum2=0;
           sum3=0;
           
           if j==c_m
           for n=((c-L1)+1):c
           sum1=sum1+T23_m1(i,n);
           sum2=sum2+T22_m1(i,n);
           sum3=sum3+T33_m1(i,n);
           end
           T23_m(i,j)=sum1/L1;
           T22_m(i,j)=sum2/L1;
           T33_m(i,j)=sum3/L1;
           continue
           end
           
           ii=((j-1)*L1)+1;%from the row with this amount by L rows should be averaged.(example:L=6, fori=2 n should vary for ((2-1)*6+1=)7 by (7+6-1=)12)
           for n=ii:(ii+L1-1)
           sum1=sum1+T23_m1(i,n);
           sum2=sum2+T22_m1(i,n);
           sum3=sum3+T33_m1(i,n);
           end
           T23_m(i,j)=sum1/L1;
           T22_m(i,j)=sum2/L1;
           T33_m(i,j)=sum3/L1;
        end 
    end
     
    
else
    c_m=c/L1;%r_m:number of rows in multilooked case
    
    T23_m=zeros(r_m,c_m);%T23_m:multilooked T23_real data
    T22_m=zeros(r_m,c_m);%T22_m:multilooked T22 data
    T33_m=zeros(r_m,c_m);%T23_m:multilooked T33 data
     
    for j=1:r_m
        sum1=0;
        sum2=0;
        sum3=0;
        for n=1:L1
            sum1=sum1+T23_m1(j,n);
            sum2=sum2+T22_m1(j,n);
            sum3=sum3+T33_m1(j,n);
        end
        T23_m(j,1)=sum1/L1;
        T22_m(j,1)=sum2/L1;
        T33_m(j,1)=sum3/L1;
    end
    for i=1:r_m
        for j=2:c_m
        sum1=0;
        sum2=0;
        sum3=0;
        ii=((j-1)*L1)+1;%from the row with this amount by L rows should be averaged.(example:L=6, fori=2 n should vary for ((2-1)*6+1=)7 by (7+6-1=)12)
        for n=ii:(ii+L1-1)
           sum1=sum1+T23_m1(i,n);
           sum2=sum2+T22_m1(i,n);
           sum3=sum3+T33_m1(i,n);
        end
           T23_m(i,j)=sum1/L1;
           T22_m(i,j)=sum2/L1;
           T33_m(i,j)=sum3/L1;
        end 
    end

end

for i=1:r_m
    for j=1:c_m
        
        teta=atan(T23_m(i,j)/(T22_m(i,j)-T33_m(i,j)));
        if teta>(pi/4)
            teta=teta-(pi/2);
        end
        if value(1)==1
            R_teta=(1+cos(2*teta))/2;
            a_m{i,j}=R_teta*a_m{i,j}*(R_teta');
        elseif value(2)==1
            R_teta=cos(2*teta);
            a_m{i,j}=R_teta*a_m{i,j}*(R_teta');
        elseif value(3)==1
            R_teta=(1+cos(2*teta))/2;
            a_m{i,j}=R_teta*a_m{i,j}*(R_teta');
        elseif value(4)==1
            R_teta=[(1+cos(2*teta))/2 (sqrt(2)*sin(2*teta))/2;(-sqrt(2)*sin(2*teta))/2 cos(2*teta)];
            a_m{i,j}=R_teta*a_m{i,j}*(R_teta');
        elseif value(5)==1
            R_teta=[cos(2*teta) (sqrt(2)*sin(2*teta))/2;(-sqrt(2)*sin(2*teta))/2 (1+cos(2*teta))/2];
            a_m{i,j}=R_teta*a_m{i,j}*(R_teta');
        elseif value(6)==1
            R_teta=[(1+cos(2*teta))/2 (sqrt(2)*sin(2*teta))/2 (1-cos(2*teta))/2;(-sqrt(2)*sin(2*teta))/2 cos(2*teta) (sqrt(2)*sin(2*teta))/2;(1-cos(2*teta))/2 (-sqrt(2)*sin(2*teta))/2 (1+cos(2*teta))/2];
            a_m{i,j}=R_teta*a_m{i,j}*(R_teta');

        end
    end
end
b_m=a_m;
% r_m
% c_m
for i=1:10
    for j=1:10
    b_m{i,j}=a_m{(i+r_m-10),(j+c_m-10)};
    b_m{i+r_m-10,j+c_m-10}=a_m{i,j};
    end
end
end
T_M=127.5*ones(r_m,c_m);
rr=(r_m-10)+1;
cc=(c_m-10)+1;
for i=1:r_m
    for j=1:c_m
                T_M(1:10,1:10)=255;  %(change class)
                T_M(rr:r_m,cc:c_m)=255; %(change class)
    end
end
%  size(a_m)
%  size(b_m)
% b_m{34,150}=[ 0.0000000000001753 + 0.0000i  -0.00086 + 0.00000055i   0.00000000000750 - 0.0002i;-0.00086 - 0.00055i   0.000000000000001004 + 0.000000i   0.000000000000000016 - 0.0071i;0.000000000000750 + 0.000002i   0.000000000000016 + 0.000071i   0.00000000000002041 + 0.000000i];
% a_m{34,150}=[1.93100 - 0.0000i  -0.0011 + 0.0117i   1.1881 - 0.0100i;1.9911 - 0.0117i   1.7787 + 0.0000i  -0.0015 - 0.0114i;1.1881 + 0.0100i  -0.0015 + 0.0114i   1.3278 - 0.0000i];


a88= a_m{8,8}
b9974=b_m{99,74}



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in checkbox1.
function checkbox1_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox1



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in radiobutton14.
function radiobutton14_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton14


% --- Executes on button press in radiobutton15.
function radiobutton15_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton15


% --- Executes on button press in radiobutton16.
function radiobutton16_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton16


% --------------------------------------------------------------------
function Untitled_1_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Untitled_4_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Untitled_5_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Untitled_7_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(handles.uipanel4,'visible','off');
set(handles.uipanel6,'visible','on');
set(handles.uipanel15,'visible','off');
set(handles.uipanel24,'visible','off');
set(handles.uipanel35,'visible','off');
set(handles.uipanel37,'visible','off');
set(handles.uipanel44,'visible','off');
% --------------------------------------------------------------------
function Untitled_6_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.uipanel4,'visible','on');
set(handles.uipanel6,'visible','off');
set(handles.uipanel15,'visible','off');
set(handles.uipanel24,'visible','off');
set(handles.uipanel35,'visible','off');
set(handles.uipanel37,'visible','off');
set(handles.uipanel44,'visible','off');
% --------------------------------------------------------------------
function Untitled_2_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Untitled_3_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.uipanel9,'visible','off');
set(handles.uipanel1,'visible','on');
set(handles.uipanel11,'visible','off');
set(handles.uipanel15,'visible','off');
set(handles.uipanel14,'visible','off');
% --------------------------------------------------------------------
function Untitled_8_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
global d a_m b_m r_m c_m Cx Cy r1 r2
L=str2double(get(handles.edit3,'String'));
value=zeros(2,1);
value(1,1)=get(handles.radiobutton17,'value');
value(2,1)=get(handles.radiobutton18,'value');
if value(1)==1
    d=2;
[a_m1,b_m1,GT]=simSWwishartMLC_forChangeDetection(L,1);
elseif value(2)==1
    d=4;
[a_m1,b_m1,GT]=simSWwishartMLC_forChangeDetection(L,0);

r1=real(r1);
r2=real(r2);
figure(25)
imagesc(imadjust(r1,stretchlim(r1)))
axis equal
axis image off;
figure(26)
imagesc(imadjust(r2,stretchlim(r2)))
axis equal
axis image off;
axes(handles.axes9);
imagesc(imadjust(r1,stretchlim(r1)))
axis image off;
axes(handles.axes10);
imagesc(imadjust(r2,stretchlim(r2)))
axis image off;
          
end


%size(a_m1)

r_m=Cx;
c_m=Cy;
a_m=cell(r_m,c_m);
b_m=cell(r_m,c_m);
for i=1:r_m
    for j=1:c_m
        a_m{i,j}=a_m1(:,:,i,j);
        b_m{i,j}=b_m1(:,:,i,j);
    end
end
% a_m{1,1}
% b_m{1,1}


function [CData1,CData2,GT]= simSWwishartMLC_forChangeDetection(L,dualpol)
% if ispc
%        path('\vahid\EO\Workspace\MATLAB\',path);
% else
%         path('/Groups/EO/Vahid/Workspace/MATLAB/',path);
% end
% 
global r1 r2 Cx Cy T_M
 Cx=250;Cy=250; %dimension
plotrgb=1;
 change=1; %1: if change happens
 sthr=2; %1: single thresholding 2:double thresholding
 a_d=0;%anomally exists if a_d==1
if dualpol
%dual pol:HH/HV
names{3} = 'Water';
params{3} = {9E-6,  [9.7, 2.5, 4.1-0.14i]};
names{2} = 'B field';
params{2} = {1E-3,  [2.8,  1.8, -0.5-0.84i]};
names{1} = 'A field';
params{1} = {6E-4,  [1.8,  2.0, 0.6-0.8i]};
names{4} = 'Forest';
params{4} = {6E-3,[0.9,  1.5, 0.3+0.1i]};
names{5} = 'G field';
params{5} = {7.1E-4, [7.7, 3.1, 4.0-1.8i]};
names{6} = 'E field';
params{6} = {5E-5, [2.2, 0.1, 1.7, 1-0.1i]};
names{7} = 'Urban';
params{7} = {4E-3, [0.83, 0.51, 2.13, -0.1+0.19i]};

for m = 1:length(params),
  mpars = params{m};
  Gm = complex(zeros(2));
  Gm(1, 1) = mpars{2}(1); %HH
  Gm(2, 2) = mpars{2}(2);%VV
  Gm(1, 2) = mpars{2}(3);
  Gm(2, 1) = conj(Gm(1, 2));
  dGm = det(Gm);
  wparams{m} = {(mpars{1}),Gm/dGm};
end;


else
    
 %quad-pol
names{3} = 'Water';
params{3} = {9E-6,  [9.7, 0.25, 2.5, 4.1-0.14i]};
names{2} = 'B field';
params{2} = {1E-3,  [2.8, 0.24, 1.8, -0.5-0.84i]};
names{1} = 'A field';
params{1} = {6E-4,  [1.8, 0.1, 2.0, 0.6-0.8i]};
names{4} = 'Forest';
params{4} = {6E-3,[0.9, 0.8, 1.5, 0.3+0.1i]};
names{5} = 'G field';
params{5} = {7.1E-4, [7.7, 0.17, 3.1, 4.0-1.8i]};
names{6} = 'E field';
params{6} = {5E-5, [2.2, 0.1, 1.7, 1-0.1i]};
names{7} = 'Urban';
params{7} = {4E-3, [0.83, 0.51, 2.13, -0.1+0.19i]};

for m = 1:length(params),
  mpars = params{m};
  Gm = complex(zeros(4));
  Gm(1, 1) = mpars{2}(1);
  Gm(2, 2) = mpars{2}(2);
  Gm(3, 3) = mpars{2}(2);
  Gm(4, 4) = mpars{2}(3);
  Gm(1, 4) = mpars{2}(4);
  Gm(4, 1) = conj(Gm(1, 4));
  dGm = det(Gm);
  wparams{m} = {(mpars{1}),Gm/dGm};
end;
end

[CData1,Mj2]= genW(L, wparams{2}{1}, wparams{2}{2}, [Cx,Cy]); %Generate Wishart data  class2 
[CData1(:,:,1:50,1:45),Mj3]= genW(L, wparams{3}{1}, wparams{3}{2}, [50,45]);   % class 3
[CData1(:,:,51:200,1:45),Mj1]=genW(L, wparams{1}{1}, wparams{1}{2}, [150,45]);  %class1
[CData1(:,:,201:250,1:45),Mj4]=genW(L, wparams{4}{1}, wparams{4}{2}, [50,45]);  %class4
[CData1(:,:,1:70,45+1:45+75),Mj7]=genW(L, wparams{7}{1}, wparams{7}{2}, [70,75]); %class7
[CData1(:,:,1:70,121:205),Mj6]=genW(L, wparams{6}{1}, wparams{6}{2}, [70,85]);   %class6
CData1(:,:,1:60,45+160+1:end)=genW(L, wparams{4}{1}, wparams{4}{2}, [60,45]);%class4
CData1(:,:,61:200,45+160+1:end)=genW(L, wparams{1}{1}, wparams{1}{2}, [140,45]);  %class1
CData1(:,:,201:end,45+160+1:end)=genW(L, wparams{3}{1}, wparams{3}{2}, [50,45]); % %class3
CData1(:,:,181:end,46:120)=genW(L, wparams{6}{1}, wparams{6}{2}, [70,75]);  %class6
[CData1(:,:,181:end,121:130),Mj5]=genW(L, wparams{5}{1},wparams{5}{2}, [70,10]);  %class5
CData1(:,:,181:end,131:205)=genW(L, wparams{7}{1}, wparams{7}{2}, [70,75]); %class7
[CData1(:,:,106:145,106:145),Mj5]=genW(L, wparams{5}{1}, wparams{5}{2}, [40,40]); %class5
[CData1(:,:,1:70,121:130),Mj5]=genW(L, wparams{5}{1}, wparams{5}{2}, [70,10]); %class5

CData1=single(CData1);
if dualpol==0
if plotrgb
    
    
    
sz=size(CData1);
d=sz(1);
if d==3  % full polarized 
  % R = 10*log10(|Shh-Svv|**2)
  % G = 10*log10(|Shv|**2)
  % B = 10*log10(|Shh+Svv|**2)

  r = 0.5*(CData1(1,1,:,:) - CData1(3,3,:,:)).^2;
  g = 2 * CData1(2,2,:,:).^2;
  b = 0.5*(CData1(1,1,:,:) + CData1(3,3,:,:)).^2;
elseif d==4
  % R = 10*log10(|Shh-Svv|**2)
  % G = 10*log10(|Shv|**2)
  % B = 10*log10(|Shh+Svv|**2)

  r = 0.5*(CData1(1,1,:,:) - CData1(4,4,:,:)).^2;
  g = 2 * CData1(2,2,:,:).^2;
  b = 0.5*(CData1(1,1,:,:) + CData1(4,4,:,:)).^2;  
else  %dual polarized
  % R = 10*log10(|Shh|**2)
  % G = 10*log10(|Shv|**2)
  % B = 10*log10(|Shh|**2)
  
  r = 0.5*(CData1(1,1,:,:).^2);
  g = 2 * CData1(2,2,:,:).^2;
  b = 0.5*(CData1(1,1,:,:).^2);
end

  % Enforce positive values
  
  r(r<eps) = eps;
  g(g<eps) = eps;
  b(b<eps) = eps;

  % Logarithmic transformation
  
  r = 10*log10(r);
  g = 10*log10(g);
  b = 10*log10(b);

  % Normalise bands to [0,1]
  
  eps2 = 10*log10(eps);
  rmin = min(r(r>eps2));
  rmax = max(r(r>eps2));
  gmin = min(g(g>eps2));
  gmax = max(g(g>eps2));
  bmin = min(b(b>eps2));
  bmax = max(b(b>eps2));

  r = (r - rmin) ./ (rmax - rmin);
  g = (g - gmin) ./ (gmax - gmin);
  b = (b - bmin) ./ (bmax - bmin);

  % Display
  
  r1 = squeeze(r);
  r1(:,:,2) = squeeze(g);
  r1(:,:,3) = squeeze(b);
  

    
    
end
end

if change==1
CData2= genW(L, wparams{2}{1}, wparams{2}{2}, [Cx,Cy]); %Generate Wishart data  class2 
CData2(:,:,1:50,1:45)= genW(L, wparams{3}{1}, wparams{3}{2}, [50,45]);   % class 3
CData2(:,:,51:200,1:45)=genW(L, wparams{1}{1}, wparams{1}{2}, [150,45]);  %class1
CData2(:,:,201:250,1:45)=genW(L, wparams{4}{1}, wparams{4}{2}, [50,45]);  %class4
CData2(:,:,1:70,45+1:45+75)=genW(L, wparams{7}{1}, wparams{7}{2}, [70,75]); %class7
CData2(:,:,1:70,121:205)=genW(L, wparams{6}{1}, wparams{6}{2}, [70,85]);   %class6
CData2(:,:,1:60,45+160+1:end)=genW(L, wparams{4}{1}, wparams{4}{2}, [60,45]);%class4
CData2(:,:,61:200,45+160+1:end)=genW(L, wparams{1}{1}, wparams{1}{2}, [140,45]);  %class1
CData2(:,:,201:end,45+160+1:end)=genW(L, wparams{3}{1}, wparams{3}{2}, [50,45]); % %class3
CData2(:,:,181:end,46:120)=genW(L, wparams{6}{1}, wparams{6}{2}, [70,75]);  %class6

if sthr==1
 CData2(:,:,181:end,121:130)=genW(L, wparams{7}{1},wparams{7}{2}, [70,10]);  %class7(change class)
 CData2(:,:,106:145,106:145)=genW(L, wparams{7}{1}, wparams{7}{2}, [40,40]); %class7 (change class)
else
[CData2(:,:,181:end,121:130)]=genW(L, wparams{7}{1},wparams{7}{2}, [70,10]);  %class7(change class)
[CData2(:,:,1:70,121:130)]=genW(L, wparams{7}{1},wparams{7}{2}, [70,10]);  %class7(change class)
[CData2(:,:,106:145,106:145)]=genW(L, wparams{6}{1}, wparams{6}{2}, [40,40]); %class1 (change class)
end
       
 CData2(:,:,181:end,131:205)=genW(L, wparams{7}{1}, wparams{7}{2}, [70,75]); %class7
 
if a_d==1
 CData2(:,:,24:25,24:26)=genW(L, wparams{1}{1},wparams{1}{2}, [2,3]);  %class1(change class)
 CData2(:,:,10:11,10:12)=genW(L, wparams{1}{1},wparams{1}{2}, [2,3]);  %class1(change class)
 CData2(:,:,35:36,15:17)=genW(L, wparams{1}{1},wparams{1}{2}, [2,3]);  %class1(change class)
 CData2(:,:,230:231,224:226)=genW(L, wparams{1}{1}, wparams{1}{2}, [2,3]); %class1 (change class)
 CData2(:,:,210:211,210:212)=genW(L, wparams{1}{1},wparams{1}{2}, [2,3]);  %class1(change class)
 CData2(:,:,240:241,214:216)=genW(L, wparams{1}{1},wparams{1}{2}, [2,3]);  %class1(change class) 
end
 
else
CData2= genW(L, wparams{2}{1}, wparams{2}{2}, [Cx,Cy]); %Generate Wishart data  class2 
CData2(:,:,1:50,1:45)= genW(L, wparams{3}{1}, wparams{3}{2}, [50,45]);   % class 3
CData2(:,:,51:200,1:45)=genW(L, wparams{1}{1}, wparams{1}{2}, [150,45]);  %class1
CData2(:,:,201:250,1:45)=genW(L, wparams{4}{1}, wparams{4}{2}, [50,45]);  %class4
CData2(:,:,1:70,45+1:45+80)=genW(L, wparams{7}{1}, wparams{7}{2}, [70,80]); %class7
CData2(:,:,1:70,126:205)=genW(L, wparams{6}{1}, wparams{6}{2}, [70,80]);   %class6
CData2(:,:,1:60,45+160+1:end)=genW(L, wparams{4}{1}, wparams{4}{2}, [60,45]);%class4
CData2(:,:,61:200,45+160+1:end)=genW(L, wparams{1}{1}, wparams{1}{2}, [140,45]);  %class1
CData2(:,:,201:end,45+160+1:end)=genW(L, wparams{3}{1}, wparams{3}{2}, [50,45]); % %class3
CData2(:,:,181:end,46:120)=genW(L, wparams{6}{1}, wparams{6}{2}, [70,75]);  %class6
CData2(:,:,181:end,121:130)=genW(L, wparams{5}{1},wparams{5}{2}, [70,10]);  %class5
CData2(:,:,181:end,131:205)=genW(L, wparams{7}{1}, wparams{7}{2}, [70,75]); %class7
CData2(:,:,106:145,106:145)=genW(L, wparams{5}{1}, wparams{5}{2}, [40,40]); %class5

if a_d==1
 CData2(:,:,24:25,24:26)=genW(L, wparams{1}{1},wparams{1}{2}, [2,3]);  %class1(change class)
 CData2(:,:,10:11,10:12)=genW(L, wparams{1}{1},wparams{1}{2}, [2,3]);  %class1(change class)
 CData2(:,:,35:36,15:17)=genW(L, wparams{1}{1},wparams{1}{2}, [2,3]);  %class1(change class)
 CData2(:,:,230:231,224:226)=genW(L, wparams{1}{1}, wparams{1}{2}, [2,3]); %class1 (change class)
 CData2(:,:,210:211,210:212)=genW(L, wparams{1}{1},wparams{1}{2}, [2,3]);  %class1(change class)
 CData2(:,:,240:241,214:216)=genW(L, wparams{1}{1},wparams{1}{2}, [2,3]);  %class1(change class)
end

end


CData2=single(CData2);

if dualpol==0
if plotrgb
    
    
    
    
sz=size(CData2);
d=sz(1);
if d==3  % full polarized 
  % R = 10*log10(|Shh-Svv|**2)
  % G = 10*log10(|Shv|**2)
  % B = 10*log10(|Shh+Svv|**2)

  r = 0.5*((CData2(1,1,:,:) - CData2(3,3,:,:)).^2);
  g = 2 * (CData2(2,2,:,:).^2);
  b = 0.5*((CData2(1,1,:,:) +CData2(3,3,:,:)).^2);
elseif d==4
  % R = 10*log10(|Shh-Svv|**2)
  % G = 10*log10(|Shv|**2)
  % B = 10*log10(|Shh+Svv|**2)

  r = 0.5*(CData2(1,1,:,:) - CData2(4,4,:,:)).^2;
  g = 2 * CData2(2,2,:,:).^2;
  b = 0.5*(CData2(1,1,:,:) + CData2(4,4,:,:)).^2;  
else  %dual polarized
  % R = 10*log10(|Shh|**2)
  % G = 10*log10(|Shv|**2)
  % B = 10*log10(|Shh|**2)
  
  r = 0.5*(CData2(1,1,:,:).^2);
  g = 2 * CData2(2,2,:,:).^2;
  b = 0.5*(CData2(1,1,:,:).^2);
end

  % Enforce positive values
  
  r(r<eps) = eps;
  g(g<eps) = eps;
  b(b<eps) = eps;

  % Logarithmic transformation
  
  r = 10*log10(r);
  g = 10*log10(g);
  b = 10*log10(b);

  % Normalise bands to [0,1]
  
  eps2 = 10*log10(eps);
  rmin = min(r(r>eps2));
  rmax = max(r(r>eps2));
  gmin = min(g(g>eps2));
  gmax = max(g(g>eps2));
  bmin = min(b(b>eps2));
  bmax = max(b(b>eps2));

  r = (r - rmin) ./ (rmax - rmin);
  g = (g - gmin) ./ (gmax - gmin);
  b = (b - bmin) ./ (bmax - bmin);

  % Display
  
  r2 = squeeze(r);
  r2(:,:,2) = squeeze(g);
  r2(:,:,3) = squeeze(b);
  

end
end
%%%%%naghsheye azmoon
T_M=127.5*ones(Cx,Cy);
for i=1:Cx
    for j=1:Cy
        if change==1
            if sthr==1
              T_M(181:end,121:130)=255;  %(change class)
              T_M(106:145,106:145)=255; %(change class)  
            elseif sthr==2
              T_M(181:end,121:130)=255;  %(change class)
              T_M(1:70,121:130)=255;  %(change class)
              T_M(106:145,106:145)=255; %(change class)
            end
        end
        if a_d==1
              T_M(24:25,24:26)=255;
              T_M(10:11,10:12)=255;
              T_M(35:36,15:17)=255;
              T_M(230:231,224:226)=255;
              T_M(210:211,210:212)=255;
              T_M(240:241,214:216)=255;
        end
    end
end


GT=2*ones(Cx, Cy); 
GT(181:end,121:130)=ones(70,10); 
GT(106:145,106:145)=ones(40,40); 
GT=uint8(GT);
figure(1);
clf
linenu = 3;fs=20;
[fA,xA] = ksdensity(Mj1,'npoints',10000);
[fB,xB] = ksdensity(Mj2,'npoints',10000);
[fC,xC] = ksdensity(Mj5,'npoints',10000);
xvector=[xA;xB;xC]';
fvector=[fA; fB; fC;]';
plot(xvector, fvector, 'LineWidth',linenu);
legend({'class-1','class-5', 'class-7'},'FontSize',fs)
xlabel(['trace(\Sigma^{-1}_jX_i)'], 'fontsize',fs)
ylabel('Probability Density', 'fontsize',fs)
set(gca,'FontSize',fs);
% trimprint(gcf, 'cc.png');



%=======================================================================
function [C,Mj] = genW(L, muz, Gam, Nvec)
      Sigma=muz*Gam;
  d = size(Sigma, 1);
  Y = sqrtm(Sigma)*complex(randn(d, prod(Nvec)*L), randn(d, prod(Nvec)*L))/sqrt(2);
  C = zeros(d, d, prod(Nvec)); Mj=zeros(1,prod(Nvec));
  for ni = 1:prod(Nvec),
    C(:, :, ni) = Y(:, (1+(ni-1)*L):(ni*L))*Y(:, (1+(ni-1)*L):(ni*L))'/L;
     Mj(ni)=abs(trace(Gam\C(:,:,ni))); %compacts to one-dimensional
  end;
  C = reshape(C, [d, d, Nvec]);

 %=======================================================================
function rgbpauliC3(C,fig)

  % Input handling
  
  if (nargin < 2)
    fig = gcf;
  end

sz=size(C);
d=sz(1);
if d==3  % full polarized 
  % R = 10*log10(|Shh-Svv|**2)
  % G = 10*log10(|Shv|**2)
  % B = 10*log10(|Shh+Svv|**2)

  r = 0.5*(C(1,1,:,:) - 2*real(C(1,3,:,:)) + C(3,3,:,:));
  g = 2 * C(2,2,:,:);
  b = 0.5*(C(1,1,:,:) + 2*real(C(1,3,:,:)) + C(3,3,:,:));
elseif d==4
  % R = 10*log10(|Shh-Svv|**2)
  % G = 10*log10(|Shv|**2)
  % B = 10*log10(|Shh+Svv|**2)

  r = 0.5*(C(1,1,:,:) - 2*real(C(1,4,:,:)) + C(4,4,:,:));
  g = 2 * C(2,2,:,:);
  b = 0.5*(C(1,1,:,:) + 2*real(C(1,4,:,:)) + C(4,4,:,:));  
else  %dual polarized
  % R = 10*log10(|Shh|**2)
  % G = 10*log10(|Shv|**2)
  % B = 10*log10(|Shh|**2)
  
  r = 0.5*(C(1,1,:,:) - 2*real(C(1,2,:,:)));
  g = 2 * C(2,2,:,:);
  b = 0.5*(C(1,1,:,:) + 2*real(C(1,2,:,:)));
end

  % Enforce positive values
  
  r(r<eps) = eps;
  g(g<eps) = eps;
  b(b<eps) = eps;

  % Logarithmic transformation
  
  r = 10*log10(r);
  g = 10*log10(g);
  b = 10*log10(b);

  % Normalise bands to [0,1]
  
  eps2 = 10*log10(eps);
  rmin = min(r(r>eps2));
  rmax = max(r(r>eps2));
  gmin = min(g(g>eps2));
  gmax = max(g(g>eps2));
  bmin = min(b(b>eps2));
  bmax = max(b(b>eps2));

  r = (r - rmin) ./ (rmax - rmin);
  g = (g - gmin) ./ (gmax - gmin);
  b = (b - bmin) ./ (bmax - bmin);

  % Display
  
  r = squeeze(r);
  r(:,:,2) = squeeze(g);
  r(:,:,3) = squeeze(b);
  figure(fig)
  
  imagesc(imadjust(r,stretchlim(r)))
   axis image off;
   %title('RGB Pauli Image');

   
 

return

function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function uipanel6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to uipanel6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called



function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double


% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit6_Callback(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit6 as text
%        str2double(get(hObject,'String')) returns contents of edit6 as a double


% --- Executes during object creation, after setting all properties.
function edit6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function pushbutton5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global d a_m b_m r_m c_m Cx Cy r1 r2
L=str2double(get(handles.edit8,'String'));
value=zeros(2,1);
value(1,1)=get(handles.radiobutton19,'value');
value(2,1)=get(handles.radiobutton20,'value');
if value(1)==1
    d=2;
[a_m1,b_m1]=simSWwishartMLC_forAnomalyDetection(L,1);
elseif value(2)==1
    d=4;
[a_m1,b_m1]=simSWwishartMLC_forAnomalyDetection(L,0);

r1=real(r1);
r2=real(r2);

axes(handles.axes11);
imagesc(imadjust(r1,stretchlim(r1)))
axis image off;
axes(handles.axes12);
imagesc(imadjust(r2,stretchlim(r2)))
axis image off;
          
end


%size(a_m1)

r_m=Cx;
c_m=Cy;
a_m=cell(r_m,c_m);
b_m=cell(r_m,c_m);
for i=1:r_m
    for j=1:c_m
        a_m{i,j}=a_m1(:,:,i,j);
        b_m{i,j}=b_m1(:,:,i,j);
    end
end




function edit8_Callback(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit8 as text
%        str2double(get(hObject,'String')) returns contents of edit8 as a double


% --- Executes during object creation, after setting all properties.
function edit8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------
function Untitled_9_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.uipanel9,'visible','on');
set(handles.uipanel1,'visible','off');
set(handles.uipanel11,'visible','off');
set(handles.uipanel15,'visible','off');
set(handles.uipanel14,'visible','off');

% --------------------------------------------------------------------
function Untitled_10_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.uipanel9,'visible','off');
set(handles.uipanel1,'visible','off');
set(handles.uipanel11,'visible','on');
set(handles.uipanel15,'visible','off');
set(handles.uipanel14,'visible','off');
function [CData1,CData2]= simSWwishartMLC_forAnomalyDetection(L,dualpol)
% if ispc
%        path('\vahid\EO\Workspace\MATLAB\',path);
% else
%         path('/Groups/EO/Vahid/Workspace/MATLAB/',path);
% end
% 
global r1 r2 Cx Cy T_M
 Cx=250;Cy=250; %dimension
plotrgb=1;
 change=1; %1: if change happens
% sthr=2; %1: single thresholding 2:double thresholding

if dualpol
%dual pol:HH/HV
names{3} = 'Water';
params{3} = {9E-6,  [9.7, 2.5, 4.1-0.14i]};
names{2} = 'B field';
params{2} = {1E-3,  [2.8,  1.8, -0.5-0.84i]};
names{1} = 'A field';
params{1} = {6E-4,  [1.8,  2.0, 0.6-0.8i]};
names{4} = 'Forest';
params{4} = {6E-3,[0.9,  1.5, 0.3+0.1i]};
names{5} = 'G field';
params{5} = {7.1E-4, [7.7, 3.1, 4.0-1.8i]};
names{6} = 'E field';
params{6} = {5E-5, [2.2, 0.1, 1.7, 1-0.1i]};
names{7} = 'Urban';
params{7} = {4E-3, [0.83, 0.51, 2.13, -0.1+0.19i]};

for m = 1:length(params),
  mpars = params{m};
  Gm = complex(zeros(2));
  Gm(1, 1) = mpars{2}(1); %HH
  Gm(2, 2) = mpars{2}(2);%VV
  Gm(1, 2) = mpars{2}(3);
  Gm(2, 1) = conj(Gm(1, 2));
  dGm = det(Gm);
  wparams{m} = {(mpars{1}),Gm/dGm};
end;


else
    
 %quad-pol
names{3} = 'Water';
params{3} = {9E-6,  [9.7, 0.25, 2.5, 4.1-0.14i]};
names{2} = 'B field';
params{2} = {1E-3,  [2.8, 0.24, 1.8, -0.5-0.84i]};
names{1} = 'A field';
params{1} = {6E-4,  [1.8, 0.1, 2.0, 0.6-0.8i]};
names{4} = 'Forest';
params{4} = {6E-3,[0.9, 0.8, 1.5, 0.3+0.1i]};
names{5} = 'G field';
params{5} = {7.1E-4, [7.7, 0.17, 3.1, 4.0-1.8i]};
names{6} = 'E field';
params{6} = {5E-5, [2.2, 0.1, 1.7, 1-0.1i]};
names{7} = 'Urban';
params{7} = {4E-3, [0.83, 0.51, 2.13, -0.1+0.19i]};

for m = 1:length(params),
  mpars = params{m};
  Gm = complex(zeros(4));
  Gm(1, 1) = mpars{2}(1);
  Gm(2, 2) = mpars{2}(2);
  Gm(3, 3) = mpars{2}(2);
  Gm(4, 4) = mpars{2}(3);
  Gm(1, 4) = mpars{2}(4);
  Gm(4, 1) = conj(Gm(1, 4));
  dGm = det(Gm);
  wparams{m} = {(mpars{1}),Gm/dGm};
end;
end

[CData1,Mj3]= genW(L, wparams{3}{1}, wparams{3}{2}, [Cx,Cy]);   %Generate Wishart data class 3
[CData1(:,:,30:31,50:52),Mj1]=genW(L, wparams{1}{1}, wparams{1}{2}, [2,3]);  %this target has been replaced in time2 data
[CData1(:,:,135,118:120),Mj1]=genW(L, wparams{1}{1}, wparams{1}{2}, [1,3]);  %this target has been replaced in time2 data
[CData1(:,:,125:127,230),Mj1]=genW(L, wparams{1}{1}, wparams{1}{2}, [3,1]);  %this target has a visible change in time2 data
[CData1(:,:,235,210:211),Mj1]=genW(L, wparams{1}{1}, wparams{1}{2}, [1,2]);  %this target has a visible change in time2 data

CData1=single(CData1);
if dualpol==0
if plotrgb
    
    
    
sz=size(CData1);
d=sz(1);
if d==3  % full polarized 
  % R = 10*log10(|Shh-Svv|**2)
  % G = 10*log10(|Shv|**2)
  % B = 10*log10(|Shh+Svv|**2)

  r = 0.5*(CData1(1,1,:,:) - 2*real(CData1(1,3,:,:)) + CData1(3,3,:,:));
  g = 2 * CData1(2,2,:,:);
  b = 0.5*(CData1(1,1,:,:) + 2*real(CData1(1,3,:,:)) + CData1(3,3,:,:));
elseif d==4
  % R = 10*log10(|Shh-Svv|**2)
  % G = 10*log10(|Shv|**2)
  % B = 10*log10(|Shh+Svv|**2)

  r = 0.5*(CData1(1,1,:,:) - 2*real(CData1(1,4,:,:)) + CData1(4,4,:,:));
  g = 2 * CData1(2,2,:,:);
  b = 0.5*(CData1(1,1,:,:) + 2*real(CData1(1,4,:,:)) + CData1(4,4,:,:));  
else  %dual polarized
  % R = 10*log10(|Shh|**2)
  % G = 10*log10(|Shv|**2)
  % B = 10*log10(|Shh|**2)
  
  r = 0.5*(CData1(1,1,:,:) - 2*real(CData1(1,2,:,:)));
  g = 2 * CData1(2,2,:,:);
  b = 0.5*(CData1(1,1,:,:) + 2*real(CData1(1,2,:,:)));
end

  % Enforce positive values
  
  r(r<eps) = eps;
  g(g<eps) = eps;
  b(b<eps) = eps;

  % Logarithmic transformation
  
  r = 10*log10(r);
  g = 10*log10(g);
  b = 10*log10(b);

  % Normalise bands to [0,1]
  
  eps2 = 10*log10(eps);
  rmin = min(r(r>eps2));
  rmax = max(r(r>eps2));
  gmin = min(g(g>eps2));
  gmax = max(g(g>eps2));
  bmin = min(b(b>eps2));
  bmax = max(b(b>eps2));

  r = (r - rmin) ./ (rmax - rmin);
  g = (g - gmin) ./ (gmax - gmin);
  b = (b - bmin) ./ (bmax - bmin);

  % Display
  
  r1 = squeeze(r);
  r1(:,:,2) = squeeze(g);
  r1(:,:,3) = squeeze(b);
  

    
    
end
end

if change==1

CData2= genW(L, wparams{3}{1}, wparams{3}{2}, [Cx,Cy]); %Generate Wishart data  class3 
%CData2(:,:,1:50,1:45)= genW(L, wparams{3}{1}, wparams{3}{2}, [50,45]);   % class 3
CData2(:,:,31:32,57:59)=genW(L, wparams{1}{1}, wparams{1}{2}, [2,3]);  %this target has been replaced in data2
CData2(:,:,70,80:82)=genW(L, wparams{1}{1}, wparams{1}{2}, [1,3]);  CData2(:,:,69,81)=genW(L, wparams{1}{1}, wparams{1}{2}, [1,1]);  %this target is inserted in time2 data
CData2(:,:,65,200:202)=genW(L, wparams{1}{1}, wparams{1}{2}, [1,3]);  %this target is inserted in time2 CDATA 
CData2(:,:,134,125:127)=genW(L, wparams{1}{1}, wparams{1}{2}, [1,3]);  %this target has been replaced
CData2(:,:,125,135:137)=genW(L, wparams{1}{1}, wparams{1}{2}, [1,3]);  %this target is inserted in time2 CDATA
CData2(:,:,230,24:27)=genW(L, wparams{1}{1}, wparams{1}{2}, [1,4]);  %this target is inserted in time2 CDATA
CData2(:,:,210,180:181)=genW(L, wparams{1}{1}, wparams{1}{2}, [1,2]);  %this target is inserted in time2 CDATA
CData2(:,:,126,229:231)=genW(L, wparams{1}{1}, wparams{1}{2}, [1,3]);  %target with visible change in comparison to time1 data
CData2(:,:,234:235,209:212)=genW(L, wparams{1}{1}, wparams{1}{2}, [2,4]);  %target with visible change

else
    
[CData1,Mj3]= genW(L, wparams{3}{1}, wparams{3}{2}, [Cx,Cy]);   %Generate Wishart data class 3
[CData1(:,:,30:31,50:52),Mj1]=genW(L, wparams{1}{1}, wparams{1}{2}, [2,3]);  %class1
[CData1(:,:,135,118:120),Mj1]=genW(L, wparams{1}{1}, wparams{1}{2}, [1,3]);  %class1
[CData1(:,:,125:127,230),Mj1]=genW(L, wparams{1}{1}, wparams{1}{2}, [3,1]);  %class1
[CData1(:,:,235,210:211),Mj1]=genW(L, wparams{1}{1}, wparams{1}{2}, [1,2]);  %class1

end


CData2=single(CData2);

if dualpol==0
if plotrgb
    
    
    
    
sz=size(CData2);
d=sz(1);
if d==3  % full polarized 
  % R = 10*log10(|Shh-Svv|**2)
  % G = 10*log10(|Shv|**2)
  % B = 10*log10(|Shh+Svv|**2)

  r = 0.5*(CData2(1,1,:,:) - 2*real(CData2(1,3,:,:)) + CData2(3,3,:,:));
  g = 2 * CData2(2,2,:,:);
  b = 0.5*(CData2(1,1,:,:) + 2*real(CData2(1,3,:,:)) + CData2(3,3,:,:));
elseif d==4
  % R = 10*log10(|Shh-Svv|**2)
  % G = 10*log10(|Shv|**2)
  % B = 10*log10(|Shh+Svv|**2)

  r = 0.5*(CData2(1,1,:,:) - 2*real(CData2(1,4,:,:)) + CData2(4,4,:,:));
  g = 2 * CData2(2,2,:,:);
  b = 0.5*(CData2(1,1,:,:) + 2*real(CData2(1,4,:,:)) + CData2(4,4,:,:));  
else  %dual polarized
  % R = 10*log10(|Shh|**2)
  % G = 10*log10(|Shv|**2)
  % B = 10*log10(|Shh|**2)
  
  r = 0.5*(CData2(1,1,:,:) - 2*real(CData2(1,2,:,:)));
  g = 2 * CData2(2,2,:,:);
  b = 0.5*(CData2(1,1,:,:) + 2*real(CData2(1,2,:,:)));
end

  % Enforce positive values
  
  r(r<eps) = eps;
  g(g<eps) = eps;
  b(b<eps) = eps;

  % Logarithmic transformation
  
  r = 10*log10(r);
  g = 10*log10(g);
  b = 10*log10(b);

  % Normalise bands to [0,1]
  
  eps2 = 10*log10(eps);
  rmin = min(r(r>eps2));
  rmax = max(r(r>eps2));
  gmin = min(g(g>eps2));
  gmax = max(g(g>eps2));
  bmin = min(b(b>eps2));
  bmax = max(b(b>eps2));

  r = (r - rmin) ./ (rmax - rmin);
  g = (g - gmin) ./ (gmax - gmin);
  b = (b - bmin) ./ (bmax - bmin);

  % Display
  
  r2 = squeeze(r);
  r2(:,:,2) = squeeze(g);
  r2(:,:,3) = squeeze(b);
  

end
end
%%%%%naghsheye azmoon
T_M=zeros(Cx,Cy);
for i=1:Cx
    for j=1:Cy
        if change==1
                T_M(30:31,50:52)=255;  %(change class)
                T_M(125:127,230)=255; %(change class)
                T_M(135,118:120)=255; %(change class)
                T_M(235,210:211)=255; %(change class)
                T_M(31:32,57:59)=255; %(change class)
                T_M(70,80:82)=255; %(change class)
                T_M(69,81)=255; %(change class)
                T_M(65,200:202)=255; %(change class)
                T_M(125,135:137)=255; %(change class)
                T_M(134,125:127)=255; %(change class)
                T_M(126,229:231)=255; %(change class)
                T_M(230,24:27)=255; %(change class)
                T_M(210,180:181)=255; %(change class)
                T_M(234:235,209:212)=255; %(change class)
                T_M(126,230)=0; %(change class)
                T_M(235,210)=0; %(change class)
                T_M(235,211)=0; %(change class)
                         
        end
    end
end


% --------------------------------------------------------------------
function Untitled_11_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.uipanel9,'visible','off');
set(handles.uipanel1,'visible','off');
set(handles.uipanel11,'visible','off');
set(handles.uipanel15,'visible','off');
set(handles.uipanel14,'visible','on');

% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global r c a b d L L1 oac_check T_M a_m b_m r_m c_m val_type;% r and c specify number of rows and columns respectively and L,L1, d, a and b  specify number of looks in Azi and range direction and size of cov matrix and cov matrix in first and 2nd images respectively. 
val_type=zeros(3,1);
val_type(1,1)=get(handles.radiobutton44,'value');
val_type(2,1)=get(handles.radiobutton45,'value');
val_type(3,1)=get(handles.radiobutton48,'value');
  if val_type(1)==1
L=str2double(get(handles.edit9,'String'));
L1=str2double(get(handles.edit10,'String'));
oac_check=get(handles.checkbox2,'value');

value=zeros(6,1);
value(1,1)=get(handles.radiobutton27,'value');
value(2,1)=get(handles.radiobutton28,'value');
value(3,1)=get(handles.radiobutton29,'value');
value(4,1)=get(handles.radiobutton30,'value');
value(5,1)=get(handles.radiobutton31,'value');
value(6,1)=get(handles.radiobutton32,'value');


if value(1)==1

    d=1;
[FileName path]=uigetfile('*.bin','Import Intensity file(time1)');
a1=multibandread([path FileName],[8767,8501,1],'int16',0,'bsq','ieee-le',{'Row',[251:1250]}, {'Column',[1251:2250]});%this is our subdata,you can use a subset by adding,{'Row',[251:8250]}, {'Column',[251:8250]}:{'Row',[100:300]}, {'Column',[110:260]}
s=size(a1);
r=s(1);
c=s(2);
[FileName path]=uigetfile('*.bin','Import Intensity file(time2)');
b2=multibandread([path FileName],[8767,8501,1],'int16',0,'bsq','ieee-le',{'Row',[251:1250]}, {'Column',[1251:2250]});%this is our subdata,you can use a subset by adding:{'Row',[100:300]}, {'Column',[110:260]}

a=cell(r,c);
b=cell(r,c);
for i=1:r
    for j=1:c
        a{i,j}=a1(i,j);
        b{i,j}=b2(i,j);
    end
end
a_m=a;
b_m=b;
r_m=r;
c_m=c;
elseif value(2)==1
    d=1;
[FileName path]=uigetfile('*.bin','Import Intensity file(time1)');
a1=multibandread([path FileName],[904,523,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
s=size(a1);
r=s(1);
c=s(2);
[FileName path]=uigetfile('*.bin','Import Intensity file(time2)',{'Row',[100:300]}, {'Column',[110:260]});
b2=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le');%this is our subdata,you can use a subset by adding:{'Row',[100:300]}, {'Column',[110:260]}
 

a=cell(r,c);
b=cell(r,c);
for i=1:r
    for j=1:c
        a{i,j}=a1(i,j);
        b{i,j}=b2(i,j);
    end
end
        
elseif value(3)==1
     d=1;   
[FileName path]=uigetfile('*.bin','Import Intensity file(time1)');
a1=multibandread([path FileName],[904,523,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
s=size(a1);
r=s(1);
c=s(2);
[FileName path]=uigetfile('*.bin','Import Intensity file(time2)');
b2=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata,you can use a subset by adding:{'Row',[100:300]}, {'Column',[110:260]}
 
a=cell(r,c);
b=cell(r,c);
for i=1:r
    for j=1:c
        a{i,j}=a1(i,j);
        b{i,j}=b2(i,j);
    end
end
        
elseif value(4)==1
    d=2;
[FileName path]=uigetfile('*.bin','Import C11(time1).bin file');
C11=multibandread([path FileName],[904,523,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata

[FileName path]=uigetfile('*.bin','Import C12_real(time1).bin file');
C12_real=multibandread([path FileName],[904,523,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
[FileName path]=uigetfile('*.bin','Import C12_imag(time1).bin file');
C12_imag=multibandread([path FileName],[904,523,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata

[FileName path]=uigetfile('*.bin','Import C22(time1).bin file');
C22=multibandread([path FileName],[904,523,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata

s=size(C11);
r=s(1);
c=s(2);

a2=zeros(r,c,4);
a2(:,:,1)=C11(:,:,1);%c11
a2(:,:,2)=C12_real(:,:,1)+1i*C12_imag(:,:,1);%c12
a2(:,:,3)=conj(a2(:,:,2));%c21
a2(:,:,4)=C22(:,:,1);%c22

[FileName path]=uigetfile('*.bin','Import C11(time2).bin file');
C11_2=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata

[FileName path]=uigetfile('*.bin','Import C12_real(time2).bin file');
C12_real_2=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
[FileName path]=uigetfile('*.bin','Import C12_imag(time2).bin file');
C12_imag_2=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata

[FileName path]=uigetfile('*.bin','Import C22(time2).bin file');
C22_2=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata

b2=zeros(r,c,4);
b2(:,:,1)=C11_2(:,:,1);%c11
b2(:,:,2)=C12_real_2(:,:,1)+1i*C12_imag_2(:,:,1);%c12
b2(:,:,3)=conj(b2(:,:,2));%c21
b2(:,:,4)=C22_2(:,:,1);%c22

a=cell(r,c);
b=cell(r,c);
for i=1:r
    for j=1:c
        a{i,j}=reshape(a2(i,j,:),2,2);
        b{i,j}=reshape(b2(i,j,:),2,2);
    end
end
        
elseif value(5)==1
    d=2;
[FileName path]=uigetfile('*.bin','Import C22(time1).bin file');
C22=multibandread([path FileName],[904,523,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata

[FileName path]=uigetfile('*.bin','Import C23_real(time1).bin file');
C23_real=multibandread([path FileName],[904,523,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
[FileName path]=uigetfile('*.bin','Import C23_imag(time1).bin file');
C23_imag=multibandread([path FileName],[904,523,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata

[FileName path]=uigetfile('*.bin','Import C33(time1).bin file');
C33=multibandread([path FileName],[904,523,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata

s=size(C22);
r=s(1);
c=s(2);

a2=zeros(r,c,4);
a2(:,:,1)=C22(:,:,1);%c11
a2(:,:,2)=C23_real(:,:,1)+1i*C23_imag(:,:,1);%c23
a2(:,:,3)=conj(a2(:,:,2));%c32
a2(:,:,4)=C33(:,:,1);%c33

[FileName path]=uigetfile('*.bin','Import C22(time1).bin file');
C22_2=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata

[FileName path]=uigetfile('*.bin','Import C23_real(time1).bin file');
C23_real_2=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
[FileName path]=uigetfile('*.bin','Import C23_imag(time1).bin file');
C23_imag_2=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata

[FileName path]=uigetfile('*.bin','Import C33(time1).bin file');
C33_2=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata

b2=zeros(r,c,4);
b2(:,:,1)=C22_2(:,:,1);%c11
b2(:,:,2)=C23_real_2(:,:,1)+1i*C23_imag_2(:,:,1);%c23
b2(:,:,3)=conj(b2(:,:,2));%c32
b2(:,:,4)=C33_2(:,:,1);%c33
a=cell(r,c);
b=cell(r,c);
for i=1:r
    for j=1:c
        a{i,j}=reshape(a2(i,j,:),2,2);
        b{i,j}=reshape(b2(i,j,:),2,2);
    end
end

elseif value(6)==1

    clear C11 C11_2 C22 C22_2 C33_3 C33 C21_2 C21 C12 C21_2 C32 C32_2 C31 C31_2 C13_2 C13 C23 C23_2
    d=3;
% [FileName path]=uigetfile('*.bin','Import C11(time1).bin file');
% C11=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
C11=multibandread('C11.bin',[1784,1031,1],'float',0,'bsq','ieee-le',{'Row',[138:795]}, {'Column',[227:676]});
% [FileName path]=uigetfile('*.bin','Import C12_real(time1).bin file');
% C12_real=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
% [FileName path]=uigetfile('*.bin','Import C12_imag(time1).bin file');
% C12_imag=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
C12_real=multibandread('C12_real.bin',[1784,1031,1],'float',0,'bsq','ieee-le',{'Row',[138:795]}, {'Column',[227:676]});
C12_imag=multibandread('C12_imag.bin',[1784,1031,1],'float',0,'bsq','ieee-le',{'Row',[138:795]}, {'Column',[227:676]});
% [FileName path]=uigetfile('*.bin','Import C13_real(time1).bin file');
% C13_real=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
% [FileName path]=uigetfile('*.bin','Import C13_imag(time1).bin file');
% C13_imag=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
C13_real=multibandread('C13_real.bin',[1784,1031,1],'float',0,'bsq','ieee-le',{'Row',[138:795]}, {'Column',[227:676]});
C13_imag=multibandread('C13_imag.bin',[1784,1031,1],'float',0,'bsq','ieee-le',{'Row',[138:795]}, {'Column',[227:676]});
% [FileName path]=uigetfile('*.bin','Import C22(time1).bin file');
% C22=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
C22=multibandread('C22.bin',[1784,1031,1],'float',0,'bsq','ieee-le',{'Row',[138:795]}, {'Column',[227:676]});
% [FileName path]=uigetfile('*.bin','Import C23_real(time1).bin file');
% C23_real=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
% [FileName path]=uigetfile('*.bin','Import C23_imag(time1).bin file');
% C23_imag=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
C23_real=multibandread('C23_real.bin',[1784,1031,1],'float',0,'bsq','ieee-le',{'Row',[138:795]}, {'Column',[227:676]});
C23_imag=multibandread('C23_imag.bin',[1784,1031,1],'float',0,'bsq','ieee-le',{'Row',[138:795]}, {'Column',[227:676]});
% [FileName path]=uigetfile('*.bin','Import C33(time1).bin file');
% C33=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
C33=multibandread('C33.bin',[1784,1031,1],'float',0,'bsq','ieee-le',{'Row',[138:795]}, {'Column',[227:676]});

s=size(C11);
r=s(1);
c=s(2);
a2=zeros(r,c,9);
a2(:,:,1)=C11(:,:,1);%c11
a2(:,:,2)=C12_real(:,:,1)+1i*C12_imag(:,:,1);%c12
a2(:,:,3)=C13_real(:,:,1)+1i*C13_imag(:,:,1);%c13
a2(:,:,4)=conj(a2(:,:,2));%c21
a2(:,:,5)=C22(:,:,1);%c22
a2(:,:,6)=C23_real(:,:,1)+1i*C23_imag(:,:,1);%c23
a2(:,:,7)=conj(a2(:,:,3));%c31
a2(:,:,8)=conj(a2(:,:,6));%c32
a2(:,:,9)=C33(:,:,1);%c33

% [FileName path]=uigetfile('*.bin','Import C11(time2).bin file');
% C11=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
C11_2=multibandread('C11_2.bin',[1770,992,1],'float',0,'bsq','ieee-le',{'Row',[132:781]}, {'Column',[160:609]});
% [FileName path]=uigetfile('*.bin','Import C12_real(time2).bin file');
% C12_real=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
% [FileName path]=uigetfile('*.bin','Import C12_imag(time2).bin file');
% C12_imag=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
C12_real_2=multibandread('C12_real_2.bin',[1770,992,1],'float',0,'bsq','ieee-le',{'Row',[132:781]}, {'Column',[160:609]});
C12_imag_2=multibandread('C12_imag_2.bin',[1770,992,1],'float',0,'bsq','ieee-le',{'Row',[132:781]}, {'Column',[160:609]});
% [FileName path]=uigetfile('*.bin','Import C13_real(time2).bin file');
% C13_real=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
% [FileName path]=uigetfile('*.bin','Import C13_imag(time2).bin file');
% C13_imag=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
C13_real_2=multibandread('C13_real_2.bin',[1770,992,1],'float',0,'bsq','ieee-le',{'Row',[132:781]}, {'Column',[160:609]});
C13_imag_2=multibandread('C13_imag_2.bin',[1770,992,1],'float',0,'bsq','ieee-le',{'Row',[132:781]}, {'Column',[160:609]});
% [FileName path]=uigetfile('*.bin','Import C22(time2).bin file');
% C22=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
C22_2=multibandread('C22_2.bin',[1770,992,1],'float',0,'bsq','ieee-le',{'Row',[132:781]}, {'Column',[160:609]});
% [FileName path]=uigetfile('*.bin','Import C23_real(time2).bin file');
% C23_real=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
% [FileName path]=uigetfile('*.bin','Import C23_imag(time2).bin file');
% C23_imag=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
C23_real_2=multibandread('C23_real_2.bin',[1770,992,1],'float',0,'bsq','ieee-le',{'Row',[132:781]}, {'Column',[160:609]});
C23_imag_2=multibandread('C23_imag_2.bin',[1770,992,1],'float',0,'bsq','ieee-le',{'Row',[132:781]}, {'Column',[160:609]});
% [FileName path]=uigetfile('*.bin','Import C33(time2).bin file');
% C33=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});%this is our subdata
C33_2=multibandread('C33_2.bin',[1770,992,1],'float',0,'bsq','ieee-le',{'Row',[132:781]}, {'Column',[160:609]});

b2=zeros(r,c,9);
b2(:,:,1)=C11_2(:,:,1);%c11
b2(:,:,2)=C12_real_2(:,:,1)+1i*C12_imag_2(:,:,1);%c12
b2(:,:,3)=C13_real_2(:,:,1)+1i*C13_imag_2(:,:,1);%c13
b2(:,:,4)=conj(b2(:,:,2));%c21
b2(:,:,5)=C22_2(:,:,1);%c22
b2(:,:,6)=C23_real_2(:,:,1)+1i*C23_imag_2(:,:,1);%c23
b2(:,:,7)=conj(b2(:,:,3));%c31
b2(:,:,8)=conj(b2(:,:,6));%c32
b2(:,:,9)=C33_2(:,:,1);%c33

a=cell(r,c);
b=cell(r,c);
for i=1:r
    for j=1:c
        a{i,j}=reshape(a2(i,j,:),3,3);
        b{i,j}=reshape(b2(i,j,:),3,3);
    end
end
    
end

% a{50,50}
% b{201,151}
%multilooking:in this section, we're gonna accomplish multilooking stage in
%the processing
%first:multilooking in azimuth direction
if (r/L)-fix(r/L)~=0
    
    r_m=1+fix(r/L);%r_m:number of rows in multilooked case
    
    a_m1=cell(r_m,c);%a_m:1st image in multilooked case
    b_m1=cell(r_m,c);%a_m:2nd image in multilooked case
 
    for j=1:c
        sum=0;
        sum2=0;
        for n=1:L
            sum=sum+a{n,j};
            sum2=sum2+b{n,j};
        end
        a_m1{1,j}=sum/L;
        b_m1{1,j}=sum2/L;
    end
    for i=2:r_m
        for j=1:c
           sum=0;
           sum2=0;
           if i==r_m
           for n=((r-L)+1):r
           sum=sum+a{n,j};
           sum2=sum2+b{n,j};
           end
           a_m1{i,j}=sum/L;
           b_m1{i,j}=sum2/L;
           continue
           end
           ii=((i-1)*L)+1;%from the row with this amount by L rows shold be averaged.(example:L=6, fori=2 n shold vary for ((2-1)*6+1=)7 by (7+6-1=)12)
           for n=ii:(ii+L-1)
           sum=sum+a{n,j};
           sum2=sum2+b{n,j};
           end
           a_m1{i,j}=sum/L;
           b_m1{i,j}=sum2/L;
        end 
    end
     
    

else
    r_m=r/L;%r_m:number of rows in multilooked case
    
    a_m1=cell(r_m,c);%a_m:1st image in multilooked case
    b_m1=cell(r_m,c);%a_m:2nd image in multilooked case
    if r_m==r
        a_m1=a;
        b_m1=b;
    else
    for j=1:c
        sum=0;
        sum2=0;
        for n=1:L
            sum=sum+a{n,j};
            sum2=sum2+b{n,j};
        end
        a_m1{1,j}=sum/L;
        b_m1{1,j}=sum2/L;
    end
    for i=2:r_m
        for j=1:c
           sum=0;
           sum2=0;
           ii=((i-1)*L)+1;%from the row with this amount by L rows should be averaged.(example:L=6, fori=2 n should vary for ((2-1)*6+1=)7 by (7+6-1=)12)
           for n=ii:(ii+L-1)
           sum=sum+a{n,j};
           sum2=sum2+b{n,j};
           end
           a_m1{i,j}=sum/L;
           b_m1{i,j}=sum2/L;
        end 
    end

    end
end


%second:multilooking in range direction
if (c/L1)-fix(c/L1)~=0
    
    c_m=1+fix(c/L1);%r_m:number of rows in multilooked case
    
    a_m=cell(r_m,c_m);%a_m:1st image in multilooked case
    b_m=cell(r_m,c_m);%a_m:2nd image in multilooked case
 
    for j=1:r_m
        sum=0;
        sum2=0;
        for n=1:L1
            sum=sum+a_m1{j,n};
            sum2=sum2+b_m1{j,n};
        end
        a_m{j,1}=sum/L1;
        b_m{j,1}=sum2/L1;
    end
    for i=1:r_m
        for j=2:c_m
           sum=0;
           sum2=0;
           if j==c_m
           for n=((c-L1)+1):c
           sum=sum+a_m1{i,n};
           sum2=sum2+b_m1{i,n};
           end
           a_m{i,j}=sum/L1;
           b_m{i,j}=sum2/L1;
           continue
           end
           ii=((j-1)*L1)+1;%from the row with this amount by L rows shold be averaged.(example:L=6, fori=2 n shold vary for ((2-1)*6+1=)7 by (7+6-1=)12)
           for n=ii:(ii+L1-1)
           sum=sum+a_m1{i,n};
           sum2=sum2+b_m1{i,n};
           end
           a_m{i,j}=sum/L1;
           b_m{i,j}=sum2/L1;
        end 
    end
     
    

else
    c_m=c/L1;%r_m:number of rows in multilooked case

    a_m=cell(r_m,c_m);%a_m:1st image in multilooked case
    b_m=cell(r_m,c_m);%a_m:2nd image in multilooked case
    if c_m==c
        a_m=a_m1;
        b_m=b_m1;
    else
    for j=1:c_m
        sum=0;
        sum2=0;
        for n=1:L1
            sum=sum+a_m1{j,n};
            sum2=sum2+b_m1{j,n};
        end
        a_m{j,1}=sum/L1;
        b_m{j,1}=sum2/L1;
    end
    for i=1:r_m
        for j=2:c_m
           sum=0;
           sum2=0;
           ii=((j-1)*L1)+1;%from the row with this amount by L rows should be averaged.(example:L=6, fori=2 n should vary for ((2-1)*6+1=)7 by (7+6-1=)12)
           for n=ii:(ii+L1-1)
           sum=sum+a_m1{i,n};
           sum2=sum2+b_m1{i,n};
           end
           a_m{i,j}=sum/L1;
           b_m{i,j}=sum2/L1;
        end 
    end

    end
end

%  size(a_m)
%  size(b_m)
 
 
% a_m{8,8}
% b_m{99,49}
%cmp=strcmp(oac_check,'Orientation angle compensation')
if oac_check==1

% [FileName path]=uigetfile('*.bin','Import T23_real(time1).bin file');
% T23_real=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});
T23_real=multibandread('T23_real.bin',[1784,1031,1],'float',0,'bsq','ieee-le',{'Row',[138:795]}, {'Column',[227:676]});
% [FileName path]=uigetfile('*.bin','Import T22(time1).bin file');
% T22=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});
T22=multibandread('T22.bin',[1784,1031,1],'float',0,'bsq','ieee-le',{'Row',[138:795]}, {'Column',[227:676]});
% [FileName path]=uigetfile('*.bin','Import T33(time1).bin file');
% T33=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});
T33=multibandread('T33.bin',[1784,1031,1],'float',0,'bsq','ieee-le',{'Row',[138:795]}, {'Column',[227:676]});


% [FileName path]=uigetfile('*.bin','Import T23_real(time2).bin file');
% T23_real=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});
T23_real_2=multibandread('T23_real_2.bin',[1770,992,1],'float',0,'bsq','ieee-le',{'Row',[132:781]}, {'Column',[160:609]});
% [FileName path]=uigetfile('*.bin','Import T22(time2).bin file');
% T22=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});
T22_2=multibandread('T22_2.bin',[1770,992,1],'float',0,'bsq','ieee-le',{'Row',[132:781]}, {'Column',[160:609]});
% [FileName path]=uigetfile('*.bin','Import T33(time2).bin file');
% T33=multibandread([path FileName],[894,501,1],'float',0,'bsq','ieee-le',{'Row',[100:300]}, {'Column',[110:260]});
T33_2=multibandread('T33_2.bin',[1770,992,1],'float',0,'bsq','ieee-le',{'Row',[132:781]}, {'Column',[160:609]});

%multilooking of T matrix data:in this section, we're gonna accomplish multilooking of T23_real, T22 and T33 in
%the processing of OAC.
if (r/L)-fix(r/L)~=0
    
    r_m=1+fix(r/L);%r_m:number of rows in multilooked case
    
    T23_m1=zeros(r_m,c);%T23_m:multilooked T23_real data
    T22_m1=zeros(r_m,c);%T22_m:multilooked T22 data
    T33_m1=zeros(r_m,c);%T23_m:multilooked T33 data
    T23_m1_2=zeros(r_m,c);%T23_m:multilooked T23_real data
    T22_m1_2=zeros(r_m,c);%T22_m:multilooked T22 data
    T33_m1_2=zeros(r_m,c);%T23_m:multilooked T33 data
    
    for j=1:c
        sum1=0;
        sum2=0;
        sum3=0;
        sum1_2=0;
        sum2_2=0;
        sum3_2=0;        
        for n=1:L
            sum1=sum1+T23_real(n,j);
            sum2=sum2+T22(n,j);
            sum3=sum3+T33(n,j);
            sum1_2=sum1_2+T23_real_2(n,j);
            sum2_2=sum2_2+T22_2(n,j);
            sum3_2=sum3_2+T33_2(n,j);            
        end
        T23_m1(1,j)=sum1/L;
        T22_m1(1,j)=sum2/L;
        T33_m1(1,j)=sum3/L;
        T23_m1_2(1,j)=sum1_2/L;
        T22_m1_2(1,j)=sum2_2/L;
        T33_m1_2(1,j)=sum3_2/L;        
    end
    for i=2:r_m
        for j=1:c
           sum1=0;
           sum2=0;
           sum3=0;
           sum1_2=0;
           sum2_2=0;
           sum3_2=0;           
           if i==r_m
           for n=((r-L)+1):r
           sum1=sum1+T23_real(n,j);
           sum2=sum2+T22(n,j);
           sum3=sum3+T33(n,j);
           sum1_2=sum1_2+T23_real_2(n,j);
           sum2_2=sum2_2+T22_2(n,j);
           sum3_2=sum3_2+T33_2(n,j);
           end
           T23_m1(i,j)=sum1/L;
           T22_m1(i,j)=sum2/L;
           T33_m1(i,j)=sum3/L;
           T23_m1_2(i,j)=sum1_2/L;
           T22_m1_2(i,j)=sum2_2/L;
           T33_m1_2(i,j)=sum3_2/L;
           continue
           end
           
           ii=((i-1)*L)+1;%from the row with this amount by L rows should be averaged.(example:L=6, fori=2 n should vary for ((2-1)*6+1=)7 by (7+6-1=)12)
           for n=ii:(ii+L-1)
           sum1=sum1+T23_real(n,j);
           sum2=sum2+T22(n,j);
           sum3=sum3+T33(n,j);
           sum1_2=sum1_2+T23_real_2(n,j);
           sum2_2=sum2_2+T22_2(n,j);
           sum3_2=sum3_2+T33_2(n,j);
           end
           T23_m1(i,j)=sum1/L;
           T22_m1(i,j)=sum2/L;
           T33_m1(i,j)=sum3/L;
           T23_m1_2(i,j)=sum1_2/L;
           T22_m1_2(i,j)=sum2_2/L;
           T33_m1_2(i,j)=sum3_2/L;
        end 
    end
     
    
else
    r_m=r/L;%r_m:number of rows in multilooked case
    
    T23_m1=zeros(r_m,c);%T23_m:multilooked T23_real data
    T22_m1=zeros(r_m,c);%T22_m:multilooked T22 data
    T33_m1=zeros(r_m,c);%T23_m:multilooked T33 data
    T23_m1_2=zeros(r_m,c);%T23_m:multilooked T23_real data
    T22_m1_2=zeros(r_m,c);%T22_m:multilooked T22 data
    T33_m1_2=zeros(r_m,c);%T23_m:multilooked T33 data     
    for j=1:c
        sum1=0;
        sum2=0;
        sum3=0;
        sum1_2=0;
        sum2_2=0;
        sum3_2=0;
        for n=1:L
            sum1=sum1+T23_real(n,j);
            sum2=sum2+T22(n,j);
            sum3=sum3+T33(n,j);
            sum1_2=sum1_2+T23_real_2(n,j);
            sum2_2=sum2_2+T22_2(n,j);
            sum3_2=sum3_2+T33_2(n,j);
        end
        T23_m1(1,j)=sum1/L;
        T22_m1(1,j)=sum2/L;
        T33_m1(1,j)=sum3/L;
        T23_m1_2(1,j)=sum1_2/L;
        T22_m1_2(1,j)=sum2_2/L;
        T33_m1_2(1,j)=sum3_2/L;
    end
    for i=2:r_m
        for j=1:c
        sum1=0;
        sum2=0;
        sum3=0;
        sum1_2=0;
        sum2_2=0;
        sum3_2=0;
        ii=((i-1)*L)+1;%from the row with this amount by L rows should be averaged.(example:L=6, fori=2 n should vary for ((2-1)*6+1=)7 by (7+6-1=)12)
        for n=ii:(ii+L-1)
           sum1=sum1+T23_real(n,j);
           sum2=sum2+T22(n,j);
           sum3=sum3+T33(n,j);
           sum1_2=sum1_2+T23_real_2(n,j);
           sum2_2=sum2_2+T22_2(n,j);
           sum3_2=sum3_2+T33_2(n,j);
        end
           T23_m1(i,j)=sum1/L;
           T22_m1(i,j)=sum2/L;
           T33_m1(i,j)=sum3/L;
           T23_m1_2(i,j)=sum1_2/L;
           T22_m1_2(i,j)=sum2_2/L;
           T33_m1_2(i,j)=sum3_2/L;
        end 
    end

end

%range directon multilooking of T matrix elements.
if (c/L1)-fix(c/L1)~=0
    
    c_m=1+fix(c/L1);%c_m:number of coloumns in multilooked case
    
    T23_m=zeros(r_m,c_m);%T23_m:multilooked T23_real data
    T22_m=zeros(r_m,c_m);%T22_m:multilooked T22 data
    T33_m=zeros(r_m,c_m);%T23_m:multilooked T33 data
    T23_m_2=zeros(r_m,c_m);%T23_m:multilooked T23_real data
    T22_m_2=zeros(r_m,c_m);%T22_m:multilooked T22 data
    T33_m_2=zeros(r_m,c_m);%T23_m:multilooked T33 data    
    for j=1:r_m
        sum1=0;
        sum2=0;
        sum3=0;
        sum1_2=0;
        sum2_2=0;
        sum3_2=0;
        for n=1:L1
            sum1=sum1+T23_m1(j,n);
            sum2=sum2+T22_m1(j,n);
            sum3=sum3+T33_m1(j,n);
            sum1_2=sum1_2+T23_m1_2(j,n);
            sum2_2=sum2_2+T22_m1_2(j,n);
            sum3_2=sum3_2+T33_m1_2(j,n);            
        end
        T23_m(j,1)=sum1/L1;
        T22_m(j,1)=sum2/L1;
        T33_m(j,1)=sum3/L1;
        T23_m_2(j,1)=sum1_2/L1;
        T22_m_2(j,1)=sum2_2/L1;
        T33_m_2(j,1)=sum3_2/L1;        
    end
    for i=1:r_m
        for j=2:c_m
           sum1=0;
           sum2=0;
           sum3=0;
           sum1_2=0;
           sum2_2=0;
           sum3_2=0;           
           if j==c_m
           for n=((c-L1)+1):c
           sum1=sum1+T23_m1(i,n);
           sum2=sum2+T22_m1(i,n);
           sum3=sum3+T33_m1(i,n);
           sum1_2=sum1_2+T23_m1_2(i,n);
           sum2_2=sum2_2+T22_m1_2(i,n);
           sum3_2=sum3_2+T33_m1_2(i,n);
           end
           T23_m(i,j)=sum1/L1;
           T22_m(i,j)=sum2/L1;
           T33_m(i,j)=sum3/L1;
           T23_m_2(i,j)=sum1_2/L1;
           T22_m_2(i,j)=sum2_2/L1;
           T33_m_2(i,j)=sum3_2/L1;
           continue
           end
           
           ii=((j-1)*L1)+1;%from the row with this amount by L rows should be averaged.(example:L=6, fori=2 n should vary for ((2-1)*6+1=)7 by (7+6-1=)12)
           for n=ii:(ii+L1-1)
           sum1=sum1+T23_m1(i,n);
           sum2=sum2+T22_m1(i,n);
           sum3=sum3+T33_m1(i,n);
           sum1_2=sum1_2+T23_m1_2(i,n);
           sum2_2=sum2_2+T22_m1_2(i,n);
           sum3_2=sum3_2+T33_m1_2(i,n);
           end
           T23_m(i,j)=sum1/L1;
           T22_m(i,j)=sum2/L1;
           T33_m(i,j)=sum3/L1;
           T23_m_2(i,j)=sum1_2/L1;
           T22_m_2(i,j)=sum2_2/L1;
           T33_m_2(i,j)=sum3_2/L1;
        end 
    end
     
    
else
    c_m=c/L1;%r_m:number of rows in multilooked case
    
    T23_m=zeros(r_m,c_m);%T23_m:multilooked T23_real data
    T22_m=zeros(r_m,c_m);%T22_m:multilooked T22 data
    T33_m=zeros(r_m,c_m);%T23_m:multilooked T33 data
    T23_m_2=zeros(r_m,c_m);%T23_m:multilooked T23_real data
    T22_m_2=zeros(r_m,c_m);%T22_m:multilooked T22 data
    T33_m_2=zeros(r_m,c_m);%T23_m:multilooked T33 data     
    for j=1:r_m
        sum1=0;
        sum2=0;
        sum3=0;
        sum1_2=0;
        sum2_2=0;
        sum3_2=0;
        for n=1:L1
            sum1=sum1+T23_m1(j,n);
            sum2=sum2+T22_m1(j,n);
            sum3=sum3+T33_m1(j,n);
            sum1_2=sum1_2+T23_m1_2(j,n);
            sum2_2=sum2_2+T22_m1_2(j,n);
            sum3_2=sum3_2+T33_m1_2(j,n);
        end
        T23_m(j,1)=sum1/L1;
        T22_m(j,1)=sum2/L1;
        T33_m(j,1)=sum3/L1;
        T23_m_2(j,1)=sum1_2/L1;
        T22_m_2(j,1)=sum2_2/L1;
        T33_m_2(j,1)=sum3_2/L1;
    end
    for i=1:r_m
        for j=2:c_m
        sum1=0;
        sum2=0;
        sum3=0;
        sum1_2=0;
        sum2_2=0;
        sum3_2=0;
        ii=((j-1)*L1)+1;%from the row with this amount by L rows should be averaged.(example:L=6, fori=2 n should vary for ((2-1)*6+1=)7 by (7+6-1=)12)
        for n=ii:(ii+L1-1)
           sum1=sum1+T23_m1(i,n);
           sum2=sum2+T22_m1(i,n);
           sum3=sum3+T33_m1(i,n);
           sum1_2=sum1_2+T23_m1_2(i,n);
           sum2_2=sum2_2+T22_m1_2(i,n);
           sum3_2=sum3_2+T33_m1_2(i,n);
        end
           T23_m(i,j)=sum1/L1;
           T22_m(i,j)=sum2/L1;
           T33_m(i,j)=sum3/L1;
           T23_m_2(i,j)=sum1_2/L1;
           T22_m_2(i,j)=sum2_2/L1;
           T33_m_2(i,j)=sum3_2/L1;
        end 
    end
%dlmwrite('T23_m.txt', T23_m, 'delimiter', '\t');
%dlmwrite('T23_m_2.txt', T23_m_2, 'delimiter', '\t');
end

for i=1:r_m
    for j=1:c_m
        
        teta=atan(T23_m(i,j)/(T22_m(i,j)-T33_m(i,j)));
        teta_2=atan(T23_m_2(i,j)/(T22_m_2(i,j)-T33_m_2(i,j)));
        if teta>(pi/4)
            teta=teta-(pi/2);
        end
        if teta_2>(pi/4)
            teta_2=teta_2-(pi/2);
        end
        if value(1)==1
            R_teta=(1+cos(2*teta))/2;
            a_m{i,j}=R_teta*a_m{i,j}*(R_teta');
            R_teta_2=(1+cos(2*teta_2))/2;
            b_m{i,j}=R_teta_2*b_m{i,j}*(R_teta_2');            
        elseif value(2)==1
            R_teta=cos(2*teta);
            a_m{i,j}=R_teta*a_m{i,j}*(R_teta');
            R_teta_2=cos(2*teta_2);
            b_m{i,j}=R_teta_2*b_m{i,j}*(R_teta_2');            
        elseif value(3)==1
            R_teta=(1+cos(2*teta))/2;
            a_m{i,j}=R_teta*a_m{i,j}*(R_teta');
            R_teta_2=(1+cos(2*teta_2))/2;
            b_m{i,j}=R_teta_2*b_m{i,j}*(R_teta_2');
        elseif value(4)==1
            R_teta=[(1+cos(2*teta))/2 (sqrt(2)*sin(2*teta))/2;(-sqrt(2)*sin(2*teta))/2 cos(2*teta)];
            a_m{i,j}=R_teta*a_m{i,j}*(R_teta');
            R_teta_2=[(1+cos(2*teta_2))/2 (sqrt(2)*sin(2*teta_2))/2;(-sqrt(2)*sin(2*teta_2))/2 cos(2*teta_2)];
            b_m{i,j}=R_teta_2*b_m{i,j}*(R_teta_2');
        elseif value(5)==1
            R_teta=[cos(2*teta) (sqrt(2)*sin(2*teta))/2;(-sqrt(2)*sin(2*teta))/2 (1+cos(2*teta))/2];
            a_m{i,j}=R_teta*a_m{i,j}*(R_teta');
            R_teta_2=[cos(2*teta_2) (sqrt(2)*sin(2*teta_2))/2;(-sqrt(2)*sin(2*teta_2))/2 (1+cos(2*teta_2))/2];
            b_m{i,j}=R_teta_2*b_m{i,j}*(R_teta_2');
        elseif value(6)==1
            R_teta=[(1+cos(2*teta))/2 (sqrt(2)*sin(2*teta))/2 (1-cos(2*teta))/2;(-sqrt(2)*sin(2*teta))/2 cos(2*teta) (sqrt(2)*sin(2*teta))/2;(1-cos(2*teta))/2 (-sqrt(2)*sin(2*teta))/2 (1+cos(2*teta))/2];
            a_m{i,j}=R_teta*a_m{i,j}*(R_teta');
            R_teta_2=[(1+cos(2*teta_2))/2 (sqrt(2)*sin(2*teta_2))/2 (1-cos(2*teta_2))/2;(-sqrt(2)*sin(2*teta_2))/2 cos(2*teta_2) (sqrt(2)*sin(2*teta_2))/2;(1-cos(2*teta_2))/2 (-sqrt(2)*sin(2*teta_2))/2 (1+cos(2*teta_2))/2];
            b_m{i,j}=R_teta_2*b_m{i,j}*(R_teta_2');
 

        end
    end
end

end
T_M=zeros(r_m,c_m);
  elseif val_type(2)==1
      d=4;
load('TwoCoregisteredMultilookedCovarianceData.mat')
s=size(C1);
r_m=s(3);
c_m=s(4);
a_m=cell(r_m,c_m);
b_m=cell(r_m,c_m);
for i=1:r_m
    for j=1:c_m
        a_m{i,j}=C1(:,:,i,j);
        b_m{i,j}=C2(:,:,i,j);
    end
end
T_M=zeros(r_m,c_m);
for i=1:r_m
    for j=1:c_m
        if GT(i,j)==1
         % if j<210 
            T_M(i,j)=255;
        %    end
        end
        if GT(i,j)==2
       %     if j<210 
            T_M(i,j)=127.5;
       %     end
        end
    end
end

  elseif val_type(3)==1
      d=4;
load('simulatedData.mat')
s=size(a_m1)
r_m=s(3)
c_m=s(4)
a_m=cell(r_m,c_m);
b_m=cell(r_m,c_m);
for i=1:r_m
    for j=1:c_m
        if T_M(i,j)==0%127.5
        T_M(i,j)=127.5;
        end
        a_m{i,j}=a_m1(:,:,i,j);
        b_m{i,j}=b_m1(:,:,i,j);
    end
end
  end
%  size(a_m)
%  size(b_m)
% b_m{34,150}=[ 0.0000000000001753 + 0.0000i  -0.00086 + 0.00000055i   0.00000000000750 - 0.0002i;-0.00086 - 0.00055i   0.000000000000001004 + 0.000000i   0.000000000000000016 - 0.0071i;0.000000000000750 + 0.000002i   0.000000000000016 + 0.000071i   0.00000000000002041 + 0.000000i];
% a_m{34,150}=[1.93100 - 0.0000i  -0.0011 + 0.0117i   1.1881 - 0.0100i;1.9911 - 0.0117i   1.7787 + 0.0000i  -0.0015 - 0.0114i;1.1881 + 0.0100i  -0.0015 + 0.0114i   1.3278 - 0.0000i];



function edit9_Callback(hObject, eventdata, handles)
% hObject    handle to edit9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit9 as text
%        str2double(get(hObject,'String')) returns contents of edit9 as a double


% --- Executes during object creation, after setting all properties.
function edit9_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in checkbox2.
function checkbox2_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox2



function edit10_Callback(hObject, eventdata, handles)
% hObject    handle to edit10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit10 as text
%        str2double(get(hObject,'String')) returns contents of edit10 as a double


% --- Executes during object creation, after setting all properties.
function edit10_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in radiobutton21.
function radiobutton21_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton21 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton21


% --- Executes on button press in radiobutton22.
function radiobutton22_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton22 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton22


% --- Executes on button press in radiobutton23.
function radiobutton23_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton23 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton23


% --- Executes on button press in radiobutton24.
function radiobutton24_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton24 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton24


% --- Executes on button press in radiobutton25.
function radiobutton25_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton25 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton25


% --- Executes on button press in radiobutton26.
function radiobutton26_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton26 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton26

% --- Executes on button press in radiobutton38.
function radiobutton38_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton38 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton38

% --- Executes on button press in radiobutton33.
function radiobutton33_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton33 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton33


% --- Executes on button press in radiobutton34.
function radiobutton34_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton34 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton34


% --- Executes on button press in radiobutton35.
function radiobutton35_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton35 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton35


% --------------------------------------------------------------------
function Untitled_12_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.uipanel9,'visible','off');
set(handles.uipanel1,'visible','off');
set(handles.uipanel11,'visible','off');
set(handles.uipanel15,'visible','on');
set(handles.uipanel14,'visible','off');
set(handles.uipanel4,'visible','off');
set(handles.uipanel6,'visible','off');
set(handles.uipanel24,'visible','off');
set(handles.uipanel35,'visible','off');
set(handles.uipanel37,'visible','off');
set(handles.uipanel44,'visible','off');
% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
value=zeros(6,1);
value(1,1)=get(handles.radiobutton33,'value');
value(2,1)=get(handles.radiobutton34,'value');
value(3,1)=get(handles.radiobutton35,'value');
format long
global s11_1 s11_2 s12_1 s12_2 s22_1 s22_2 k1 k2 w1 w2 T6
if value(1)==1 
    w1=[(1/sqrt(2));(1/sqrt(2));0];
    w2=[(1/sqrt(2));(1/sqrt(2));0];
[FileName path]=uigetfile('*.bin','Import s11_time1.bin file');
a11_1=multibandread([path FileName],[18432,1088,2],'float',0,'bsq','ieee-le');%this is our subdata of Shh_time1

[FileName path]=uigetfile('*.bin','Import s11_time2.bin file');
a11_2=multibandread([path FileName],[18432,1088,2],'float',0,'bsq','ieee-le');%this is our subdata of Shh_time2


ba1=a11_1(:,:,1)';ca1=a11_1(:,:,2)';
da1=ba1(:);ea1=ca1(:);
fa1=[da1;ea1];
ba2=a11_2(:,:,1)';ca2=a11_2(:,:,2)';
da2=ba2(:);ea2=ca2(:);
fa2=[da2;ea2];
co=0;
ga1=zeros(size(a11_1,1)*size(a11_1,2),1);
ga2=zeros(size(a11_1,1)*size(a11_1,2),1);
for i=1:2:size(a11_1,3)*size(a11_1,1)*size(a11_1,2)
    co = co + 1;
    ga1(co,1) = fa1(i);
    ga1(co,2) = fa1(i+1);
    ga2(co,1) = fa2(i);
    ga2(co,2) = fa2(i+1);    
end
rs11_1=zeros(size(a11_1,1),size(a11_1,2));
rs11_2=zeros(size(a11_2,1),size(a11_2,2));
cou=0;
for i=1:size(a11_1,1)
    for j=1:size(a11_1,2)
     cou=cou+1;
     rs11_1(i,j)=complex(ga1(cou,1),ga1(cou,2));
     rs11_2(i,j)=complex(ga2(cou,1),ga2(cou,2));
    end
end
s11_1=rs11_1(60:2059,1:1000);
s11_2=rs11_2(1:2000,70:1069);
clear a11_1 a11_2 ba1 ba2 ca1 ca2 da1 da2 ea1 ea2 fa1 fa2 ga1 ga2 rs11_1 rs11_2
[FileName path]=uigetfile('*.bin','Import s12_time1.bin file');
a12_1=multibandread([path FileName],[18432,1088,2],'float',0,'bsq','ieee-le');%this is our subdata of Shh_time1
[FileName path]=uigetfile('*.bin','Import s12_time2.bin file');
a12_2=multibandread([path FileName],[18432,1088,2],'float',0,'bsq','ieee-le');%this is our subdata of Shh_time2


ba1=a12_1(:,:,1)';ca1=a12_1(:,:,2)';
da1=ba1(:);ea1=ca1(:);
fa1=[da1;ea1];
ba2=a12_2(:,:,1)';ca2=a12_2(:,:,2)';
da2=ba2(:);ea2=ca2(:);
fa2=[da2;ea2];
co=0;
ga1=zeros(size(a12_1,1)*size(a12_1,2),1);
ga2=zeros(size(a12_1,1)*size(a12_1,2),1);
for i=1:2:size(a12_1,3)*size(a12_1,1)*size(a12_1,2)
    co = co + 1;
    ga1(co,1) = fa1(i);
    ga1(co,2) = fa1(i+1);
    ga2(co,1) = fa2(i);
    ga2(co,2) = fa2(i+1);    
end
rs12_1=zeros(size(a12_1,1),size(a12_1,2));
rs12_2=zeros(size(a12_2,1),size(a12_2,2));
cou=0;
for i=1:size(a12_1,1)
    for j=1:size(a12_1,2)
     cou=cou+1;
     rs12_1(i,j)=complex(ga1(cou,1),ga1(cou,2));
     rs12_2(i,j)=complex(ga2(cou,1),ga2(cou,2));
    end
end
s12_1=rs12_1(60:2059,1:1000);
s12_2=rs12_2(1:2000,70:1069);
clear a12_1 a12_2 ba1 ba2 ca1 ca2 da1 da2 ea1 ea2 fa1 fa2 ga1 ga2 rs12_1 rs12_2
[FileName path]=uigetfile('*.bin','Import s22_time1.bin file');
a22_1=multibandread([path FileName],[18432,1088,2],'float',0,'bsq','ieee-le');%this is our subdata of Shh_time1
[FileName path]=uigetfile('*.bin','Import s22_time2.bin file');
a22_2=multibandread([path FileName],[18432,1088,2],'float',0,'bsq','ieee-le');%this is our subdata of Shh_time2


ba1=a22_1(:,:,1)';ca1=a22_1(:,:,2)';
da1=ba1(:);ea1=ca1(:);
fa1=[da1;ea1];
ba2=a22_2(:,:,1)';ca2=a22_2(:,:,2)';
da2=ba2(:);ea2=ca2(:);
fa2=[da2;ea2];
co=0;
ga1=zeros(size(a22_1,1)*size(a22_1,2),1);
ga2=zeros(size(a22_1,1)*size(a22_1,2),1);
for i=1:2:size(a22_1,3)*size(a22_1,1)*size(a22_1,2)
    co = co + 1;
    ga1(co,1) = fa1(i);
    ga1(co,2) = fa1(i+1);
    ga2(co,1) = fa2(i);
    ga2(co,2) = fa2(i+1);    
end
rs22_1=zeros(size(a22_1,1),size(a22_1,2));
rs22_2=zeros(size(a22_2,1),size(a22_2,2));
cou=0;
for i=1:size(a22_1,1)
    for j=1:size(a22_1,2)
     cou=cou+1;
     rs22_1(i,j)=complex(ga1(cou,1),ga1(cou,2));
     rs22_2(i,j)=complex(ga2(cou,1),ga2(cou,2));
    end
end
s22_1=rs22_1(60:2059,1:1000);
s22_2=rs22_2(1:2000,70:1069);
clear a22_1 a22_2 ba1 ba2 ca1 ca2 da1 da2 ea1 ea2 fa1 fa2 ga1 ga2 rs22_1 rs22_2
k1=cell(size(s11_1,1),size(s11_1,2));
k2=cell(size(s11_2,1),size(s11_2,2));
T6=cell(size(s11_1,1),size(s11_1,2));
rgb1=zeros(size(s11_1,1),size(s11_1,2),3);%RGB Pauli
rgb2=zeros(size(s11_2,1),size(s11_2,2),3);%RGB Pauli
for i=1:size(s11_1,1)
    for j=1:size(s11_1,2)
    k1{i,j}=(1/sqrt(2))*[s11_1(i,j)+s22_1(i,j);s11_1(i,j)-s22_1(i,j);2*s12_1(i,j)];
    k2{i,j}=(1/sqrt(2))*[s11_2(i,j)+s22_2(i,j);s11_2(i,j)-s22_2(i,j);2*s12_2(i,j)];
    T6{i,j}=[k1{i,j}*k1{i,j}' k1{i,j}*k2{i,j}';k2{i,j}*k1{i,j}' k2{i,j}*k2{i,j}'];
    rgb1(i,j,1)=T6{i,j}(2,2);
    rgb1(i,j,2)=T6{i,j}(3,3);
    rgb1(i,j,3)=T6{i,j}(1,1);
    rgb2(i,j,1)=T6{i,j}(5,5);
    rgb2(i,j,2)=T6{i,j}(6,6);
    rgb2(i,j,3)=T6{i,j}(4,4);    
    end
end
 rgb2(rgb2<eps)=eps;
 rgb2=10*log10(rgb2);
 rgb1(rgb1<eps)=eps;
 rgb1=10*log10(rgb1);

 
r2min = min(min(rgb2(:,:,1)));
r2max = max(max(rgb2(:,:,1)));
g2min = min(min(rgb2(:,:,2)));
g2max = max(max(rgb2(:,:,2)));
b2min = min(min(rgb2(:,:,3)));
b2max = max(max(rgb2(:,:,3)));

r1min = min(min(rgb1(:,:,1)));
r1max = max(max(rgb1(:,:,1)));
g1min = min(min(rgb1(:,:,2)));
g1max = max(max(rgb1(:,:,2)));
b1min = min(min(rgb1(:,:,3)));
b1max = max(max(rgb1(:,:,3)));
  

rgb1(:,:,1) = (rgb1(:,:,1) - r1min) ./ (r1max - r1min);
rgb1(:,:,2) = (rgb1(:,:,2) - g1min) ./ (g1max - g1min);
rgb1(:,:,3) = (rgb1(:,:,3) - b1min) ./ (b1max - b1min);
rgb2(:,:,1) = (rgb2(:,:,1) - r2min) ./ (r2max - r2min);
rgb2(:,:,2) = (rgb2(:,:,2) - g2min) ./ (g2max - g2min);
rgb2(:,:,3) = (rgb2(:,:,3) - b2min) ./ (b2max - b2min);

axes(handles.axes17);
imagesc(imadjust(rgb1,stretchlim(rgb1)))
axis image off;
axes(handles.axes18);
imagesc(imadjust(rgb2,stretchlim(rgb2)))
axis image off;
clear rgb1 rgb2

elseif value(2)==1
    w1=[0;0;(1/sqrt(2))];
    w2=[0;0;(1/sqrt(2))];
[FileName path]=uigetfile('*.bin','Import s11_time1.bin file');
a11_1=multibandread([path FileName],[18432,1088,2],'float',0,'bsq','ieee-le');%this is our subdata of Shh_time1

[FileName path]=uigetfile('*.bin','Import s11_time2.bin file');
a11_2=multibandread([path FileName],[18432,1088,2],'float',0,'bsq','ieee-le');%this is our subdata of Shh_time2


ba1=a11_1(:,:,1)';ca1=a11_1(:,:,2)';
da1=ba1(:);ea1=ca1(:);
fa1=[da1;ea1];
ba2=a11_2(:,:,1)';ca2=a11_2(:,:,2)';
da2=ba2(:);ea2=ca2(:);
fa2=[da2;ea2];
co=0;
ga1=zeros(size(a11_1,1)*size(a11_1,2),1);
ga2=zeros(size(a11_1,1)*size(a11_1,2),1);
for i=1:2:size(a11_1,3)*size(a11_1,1)*size(a11_1,2)
    co = co + 1;
    ga1(co,1) = fa1(i);
    ga1(co,2) = fa1(i+1);
    ga2(co,1) = fa2(i);
    ga2(co,2) = fa2(i+1);    
end
rs11_1=zeros(size(a11_1,1),size(a11_1,2));
rs11_2=zeros(size(a11_2,1),size(a11_2,2));
cou=0;
for i=1:size(a11_1,1)
    for j=1:size(a11_1,2)
     cou=cou+1;
     rs11_1(i,j)=complex(ga1(cou,1),ga1(cou,2));
     rs11_2(i,j)=complex(ga2(cou,1),ga2(cou,2));
    end
end
s11_1=rs11_1(60:2059,1:1000);
s11_2=rs11_2(1:2000,70:1069);
clear a11_1 a11_2 ba1 ba2 ca1 ca2 da1 da2 ea1 ea2 fa1 fa2 ga1 ga2 rs11_1 rs11_2
[FileName path]=uigetfile('*.bin','Import s12_time1.bin file');
a12_1=multibandread([path FileName],[18432,1088,2],'float',0,'bsq','ieee-le');%this is our subdata of Shh_time1
[FileName path]=uigetfile('*.bin','Import s12_time2.bin file');
a12_2=multibandread([path FileName],[18432,1088,2],'float',0,'bsq','ieee-le');%this is our subdata of Shh_time2


ba1=a12_1(:,:,1)';ca1=a12_1(:,:,2)';
da1=ba1(:);ea1=ca1(:);
fa1=[da1;ea1];
ba2=a12_2(:,:,1)';ca2=a12_2(:,:,2)';
da2=ba2(:);ea2=ca2(:);
fa2=[da2;ea2];
co=0;
ga1=zeros(size(a12_1,1)*size(a12_1,2),1);
ga2=zeros(size(a12_1,1)*size(a12_1,2),1);
for i=1:2:size(a12_1,3)*size(a12_1,1)*size(a12_1,2)
    co = co + 1;
    ga1(co,1) = fa1(i);
    ga1(co,2) = fa1(i+1);
    ga2(co,1) = fa2(i);
    ga2(co,2) = fa2(i+1);    
end
rs12_1=zeros(size(a12_1,1),size(a12_1,2));
rs12_2=zeros(size(a12_2,1),size(a12_2,2));
cou=0;
for i=1:size(a12_1,1)
    for j=1:size(a12_1,2)
     cou=cou+1;
     rs12_1(i,j)=complex(ga1(cou,1),ga1(cou,2));
     rs12_2(i,j)=complex(ga2(cou,1),ga2(cou,2));
    end
end
s12_1=rs12_1(60:2059,1:1000);
s12_2=rs12_2(1:2000,70:1069);
clear a12_1 a12_2 ba1 ba2 ca1 ca2 da1 da2 ea1 ea2 fa1 fa2 ga1 ga2 rs12_1 rs12_2
[FileName path]=uigetfile('*.bin','Import s22_time1.bin file');
a22_1=multibandread([path FileName],[18432,1088,2],'float',0,'bsq','ieee-le');%this is our subdata of Shh_time1
[FileName path]=uigetfile('*.bin','Import s22_time2.bin file');
a22_2=multibandread([path FileName],[18432,1088,2],'float',0,'bsq','ieee-le');%this is our subdata of Shh_time2


ba1=a22_1(:,:,1)';ca1=a22_1(:,:,2)';
da1=ba1(:);ea1=ca1(:);
fa1=[da1;ea1];
ba2=a22_2(:,:,1)';ca2=a22_2(:,:,2)';
da2=ba2(:);ea2=ca2(:);
fa2=[da2;ea2];
co=0;
ga1=zeros(size(a22_1,1)*size(a22_1,2),1);
ga2=zeros(size(a22_1,1)*size(a22_1,2),1);
for i=1:2:size(a22_1,3)*size(a22_1,1)*size(a22_1,2)
    co = co + 1;
    ga1(co,1) = fa1(i);
    ga1(co,2) = fa1(i+1);
    ga2(co,1) = fa2(i);
    ga2(co,2) = fa2(i+1);    
end
rs22_1=zeros(size(a22_1,1),size(a22_1,2));
rs22_2=zeros(size(a22_2,1),size(a22_2,2));
cou=0;
for i=1:size(a22_1,1)
    for j=1:size(a22_1,2)
     cou=cou+1;
     rs22_1(i,j)=complex(ga1(cou,1),ga1(cou,2));
     rs22_2(i,j)=complex(ga2(cou,1),ga2(cou,2));
    end
end
s22_1=rs22_1(60:2059,1:1000);
s22_2=rs22_2(1:2000,70:1069);
clear a22_1 a22_2 ba1 ba2 ca1 ca2 da1 da2 ea1 ea2 fa1 fa2 ga1 ga2 rs22_1 rs22_2
k1=cell(size(s11_1,1),size(s11_1,2));
k2=cell(size(s11_2,1),size(s11_2,2));
T6=cell(size(s11_1,1),size(s11_1,2));
rgb1=zeros(size(s11_1,1),size(s11_1,2),3);%RGB Pauli
rgb2=zeros(size(s11_2,1),size(s11_2,2),3);%RGB Pauli
for i=1:size(s11_1,1)
    for j=1:size(s11_1,2)
    k1{i,j}=(1/sqrt(2))*[s11_1(i,j)+s22_1(i,j);s11_1(i,j)-s22_1(i,j);2*s12_1(i,j)];
    k2{i,j}=(1/sqrt(2))*[s11_2(i,j)+s22_2(i,j);s11_2(i,j)-s22_2(i,j);2*s12_2(i,j)];
    T6{i,j}=[k1{i,j}*k1{i,j}' k1{i,j}*k2{i,j}';k2{i,j}*k1{i,j}' k2{i,j}*k2{i,j}'];
    rgb1(i,j,1)=T6{i,j}(2,2);
    rgb1(i,j,2)=T6{i,j}(3,3);
    rgb1(i,j,3)=T6{i,j}(1,1);
    rgb2(i,j,1)=T6{i,j}(5,5);
    rgb2(i,j,2)=T6{i,j}(6,6);
    rgb2(i,j,3)=T6{i,j}(4,4);    
    end
end
 rgb2(rgb2<eps)=eps;
 rgb2=10*log10(rgb2);
 rgb1(rgb1<eps)=eps;
 rgb1=10*log10(rgb1);

 
r2min = min(min(rgb2(:,:,1)));
r2max = max(max(rgb2(:,:,1)));
g2min = min(min(rgb2(:,:,2)));
g2max = max(max(rgb2(:,:,2)));
b2min = min(min(rgb2(:,:,3)));
b2max = max(max(rgb2(:,:,3)));

r1min = min(min(rgb1(:,:,1)));
r1max = max(max(rgb1(:,:,1)));
g1min = min(min(rgb1(:,:,2)));
g1max = max(max(rgb1(:,:,2)));
b1min = min(min(rgb1(:,:,3)));
b1max = max(max(rgb1(:,:,3)));
  

rgb1(:,:,1) = (rgb1(:,:,1) - r1min) ./ (r1max - r1min);
rgb1(:,:,2) = (rgb1(:,:,2) - g1min) ./ (g1max - g1min);
rgb1(:,:,3) = (rgb1(:,:,3) - b1min) ./ (b1max - b1min);
rgb2(:,:,1) = (rgb2(:,:,1) - r2min) ./ (r2max - r2min);
rgb2(:,:,2) = (rgb2(:,:,2) - g2min) ./ (g2max - g2min);
rgb2(:,:,3) = (rgb2(:,:,3) - b2min) ./ (b2max - b2min);

axes(handles.axes17);
imagesc(imadjust(rgb1,stretchlim(rgb1)))
axis image off;
axes(handles.axes18);
imagesc(imadjust(rgb2,stretchlim(rgb2)))
axis image off;
clear rgb1 rgb2

elseif value(3)==1
    w1=[(1/sqrt(2));(-1/sqrt(2));0];
    w2=[(1/sqrt(2));(-1/sqrt(2));0];
[FileName path]=uigetfile('*.bin','Import s11_time1.bin file');
a11_1=multibandread([path FileName],[18432,1088,2],'float',0,'bsq','ieee-le');%this is our subdata of Shh_time1

[FileName path]=uigetfile('*.bin','Import s11_time2.bin file');
a11_2=multibandread([path FileName],[18432,1088,2],'float',0,'bsq','ieee-le');%this is our subdata of Shh_time2


ba1=a11_1(:,:,1)';ca1=a11_1(:,:,2)';
da1=ba1(:);ea1=ca1(:);
fa1=[da1;ea1];
ba2=a11_2(:,:,1)';ca2=a11_2(:,:,2)';
da2=ba2(:);ea2=ca2(:);
fa2=[da2;ea2];
co=0;
ga1=zeros(size(a11_1,1)*size(a11_1,2),1);
ga2=zeros(size(a11_1,1)*size(a11_1,2),1);
for i=1:2:size(a11_1,3)*size(a11_1,1)*size(a11_1,2)
    co = co + 1;
    ga1(co,1) = fa1(i);
    ga1(co,2) = fa1(i+1);
    ga2(co,1) = fa2(i);
    ga2(co,2) = fa2(i+1);    
end
rs11_1=zeros(size(a11_1,1),size(a11_1,2));
rs11_2=zeros(size(a11_2,1),size(a11_2,2));
cou=0;
for i=1:size(a11_1,1)
    for j=1:size(a11_1,2)
     cou=cou+1;
     rs11_1(i,j)=complex(ga1(cou,1),ga1(cou,2));
     rs11_2(i,j)=complex(ga2(cou,1),ga2(cou,2));
    end
end
s11_1=rs11_1(60:2059,1:1000);
s11_2=rs11_2(1:2000,70:1069);
clear a11_1 a11_2 ba1 ba2 ca1 ca2 da1 da2 ea1 ea2 fa1 fa2 ga1 ga2 rs11_1 rs11_2
[FileName path]=uigetfile('*.bin','Import s12_time1.bin file');
a12_1=multibandread([path FileName],[18432,1088,2],'float',0,'bsq','ieee-le');%this is our subdata of Shh_time1
[FileName path]=uigetfile('*.bin','Import s12_time2.bin file');
a12_2=multibandread([path FileName],[18432,1088,2],'float',0,'bsq','ieee-le');%this is our subdata of Shh_time2


ba1=a12_1(:,:,1)';ca1=a12_1(:,:,2)';
da1=ba1(:);ea1=ca1(:);
fa1=[da1;ea1];
ba2=a12_2(:,:,1)';ca2=a12_2(:,:,2)';
da2=ba2(:);ea2=ca2(:);
fa2=[da2;ea2];
co=0;
ga1=zeros(size(a12_1,1)*size(a12_1,2),1);
ga2=zeros(size(a12_1,1)*size(a12_1,2),1);
for i=1:2:size(a12_1,3)*size(a12_1,1)*size(a12_1,2)
    co = co + 1;
    ga1(co,1) = fa1(i);
    ga1(co,2) = fa1(i+1);
    ga2(co,1) = fa2(i);
    ga2(co,2) = fa2(i+1);    
end
rs12_1=zeros(size(a12_1,1),size(a12_1,2));
rs12_2=zeros(size(a12_2,1),size(a12_2,2));
cou=0;
for i=1:size(a12_1,1)
    for j=1:size(a12_1,2)
     cou=cou+1;
     rs12_1(i,j)=complex(ga1(cou,1),ga1(cou,2));
     rs12_2(i,j)=complex(ga2(cou,1),ga2(cou,2));
    end
end
s12_1=rs12_1(60:2059,1:1000);
s12_2=rs12_2(1:2000,70:1069);
clear a12_1 a12_2 ba1 ba2 ca1 ca2 da1 da2 ea1 ea2 fa1 fa2 ga1 ga2 rs12_1 rs12_2
[FileName path]=uigetfile('*.bin','Import s22_time1.bin file');
a22_1=multibandread([path FileName],[18432,1088,2],'float',0,'bsq','ieee-le');%this is our subdata of Shh_time1
[FileName path]=uigetfile('*.bin','Import s22_time2.bin file');
a22_2=multibandread([path FileName],[18432,1088,2],'float',0,'bsq','ieee-le');%this is our subdata of Shh_time2


ba1=a22_1(:,:,1)';ca1=a22_1(:,:,2)';
da1=ba1(:);ea1=ca1(:);
fa1=[da1;ea1];
ba2=a22_2(:,:,1)';ca2=a22_2(:,:,2)';
da2=ba2(:);ea2=ca2(:);
fa2=[da2;ea2];
co=0;
ga1=zeros(size(a22_1,1)*size(a22_1,2),1);
ga2=zeros(size(a22_1,1)*size(a22_1,2),1);
for i=1:2:size(a22_1,3)*size(a22_1,1)*size(a22_1,2)
    co = co + 1;
    ga1(co,1) = fa1(i);
    ga1(co,2) = fa1(i+1);
    ga2(co,1) = fa2(i);
    ga2(co,2) = fa2(i+1);    
end
rs22_1=zeros(size(a22_1,1),size(a22_1,2));
rs22_2=zeros(size(a22_2,1),size(a22_2,2));
cou=0;
for i=1:size(a22_1,1)
    for j=1:size(a22_1,2)
     cou=cou+1;
     rs22_1(i,j)=complex(ga1(cou,1),ga1(cou,2));
     rs22_2(i,j)=complex(ga2(cou,1),ga2(cou,2));
    end
end
s22_1=rs22_1(60:2059,1:1000);
s22_2=rs22_2(1:2000,70:1069);
clear a22_1 a22_2 ba1 ba2 ca1 ca2 da1 da2 ea1 ea2 fa1 fa2 ga1 ga2 rs22_1 rs22_2
k1=cell(size(s11_1,1),size(s11_1,2));
k2=cell(size(s11_2,1),size(s11_2,2));
T6=cell(size(s11_1,1),size(s11_1,2));
rgb1=zeros(size(s11_1,1),size(s11_1,2),3);%RGB Pauli
rgb2=zeros(size(s11_2,1),size(s11_2,2),3);%RGB Pauli
for i=1:size(s11_1,1)
    for j=1:size(s11_1,2)
    k1{i,j}=(1/sqrt(2))*[s11_1(i,j)+s22_1(i,j);s11_1(i,j)-s22_1(i,j);2*s12_1(i,j)];
    k2{i,j}=(1/sqrt(2))*[s11_2(i,j)+s22_2(i,j);s11_2(i,j)-s22_2(i,j);2*s12_2(i,j)];
    T6{i,j}=[k1{i,j}*k1{i,j}' k1{i,j}*k2{i,j}';k2{i,j}*k1{i,j}' k2{i,j}*k2{i,j}'];
    rgb1(i,j,1)=T6{i,j}(2,2);
    rgb1(i,j,2)=T6{i,j}(3,3);
    rgb1(i,j,3)=T6{i,j}(1,1);
    rgb2(i,j,1)=T6{i,j}(5,5);
    rgb2(i,j,2)=T6{i,j}(6,6);
    rgb2(i,j,3)=T6{i,j}(4,4);    
    end
end
 rgb2(rgb2<eps)=eps;
 rgb2=10*log10(rgb2);
 rgb1(rgb1<eps)=eps;
 rgb1=10*log10(rgb1);

 
r2min = min(min(rgb2(:,:,1)));
r2max = max(max(rgb2(:,:,1)));
g2min = min(min(rgb2(:,:,2)));
g2max = max(max(rgb2(:,:,2)));
b2min = min(min(rgb2(:,:,3)));
b2max = max(max(rgb2(:,:,3)));

r1min = min(min(rgb1(:,:,1)));
r1max = max(max(rgb1(:,:,1)));
g1min = min(min(rgb1(:,:,2)));
g1max = max(max(rgb1(:,:,2)));
b1min = min(min(rgb1(:,:,3)));
b1max = max(max(rgb1(:,:,3)));
  

rgb1(:,:,1) = (rgb1(:,:,1) - r1min) ./ (r1max - r1min);
rgb1(:,:,2) = (rgb1(:,:,2) - g1min) ./ (g1max - g1min);
rgb1(:,:,3) = (rgb1(:,:,3) - b1min) ./ (b1max - b1min);
rgb2(:,:,1) = (rgb2(:,:,1) - r2min) ./ (r2max - r2min);
rgb2(:,:,2) = (rgb2(:,:,2) - g2min) ./ (g2max - g2min);
rgb2(:,:,3) = (rgb2(:,:,3) - b2min) ./ (b2max - b2min);

axes(handles.axes17);
imagesc(imadjust(rgb1,stretchlim(rgb1)))
axis image off;
axes(handles.axes18);
imagesc(imadjust(rgb2,stretchlim(rgb2)))
axis image off;
clear rgb1 rgb2
end


Gamma=zeros(size(s11_1,1),size(s11_1,2));%Polarimetric Interferometric coherence $$\gamma \left(w_1,w_2 \right)$$
for i=1:size(s11_1,1)
    for j=1:size(s11_1,2)
        Gamma(i,j)=w1'*T6{i,j}(1:3,4:6)*w2/sqrt((w1'*T6{i,j}(1:3,1:3)*w1)*(w2'*T6{i,j}(4:6,4:6)*w2));
    end
end
%size(T6)
%T6{1,1}
%Gamma(1,1)
Gamma=abs(Gamma);
%dlmwrite('complexPolInSAR_coherence2.txt', Gamma, 'delimiter', '\t');
Gamma(Gamma<eps)=eps;Gamma=10*log10(Gamma);
migamma=min(Gamma(Gamma>eps));
magamma=max(Gamma(Gamma>eps));
Gamma=(Gamma-migamma)./(magamma-migamma);

axes(handles.axes19);
imagesc(imadjust(Gamma,stretchlim(Gamma)))
axis image off;


% --- Executes during object creation, after setting all properties.
function uipanel5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to uipanel5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called



function edit13_Callback(hObject, eventdata, handles)
% hObject    handle to edit13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit13 as text
%        str2double(get(hObject,'String')) returns contents of edit13 as a double


% --- Executes during object creation, after setting all properties.
function edit13_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit12_Callback(hObject, eventdata, handles)
% hObject    handle to edit12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit12 as text
%        str2double(get(hObject,'String')) returns contents of edit12 as a double


% --- Executes during object creation, after setting all properties.
function edit12_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit11_Callback(hObject, eventdata, handles)
% hObject    handle to edit11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit11 as text
%        str2double(get(hObject,'String')) returns contents of edit11 as a double


% --- Executes during object creation, after setting all properties.
function edit11_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton11.
function pushbutton11_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global t_s d ts L r r_m c_m T_M TS TSS k2 k3
format long 
gau_s=str2double(get(handles.edit11,'String'));%gaurd area size
sg=(gau_s-1)/2;
tar_s=str2double(get(handles.edit12,'String'));%Target area size
st=(tar_s-1)/2;
if fix(gau_s/2)==(gau_s/2) || fix(tar_s/2)==(tar_s/2) || gau_s<=tar_s%in case the Gaurd size or Target size is not a even number or the Gaurd size is smaller than the Target size
    msgbox('Try another correct values for Sliding Window sizes!','not suitable SW sizes','error');
    return
end

pfa=str2double(get(handles.edit13,'String'));%Probability of False Alarm
value=zeros(2,1);
value(1,1)=get(handles.radiobutton49,'value');%matching PE method
value(2,1)=get(handles.radiobutton50,'value');%log cumulant PE method

val_method=zeros(2,1);
val_method(1,1)=get(handles.radiobutton51,'value');%applying CFAR on ts image
val_method(2,1)=get(handles.radiobutton52,'value');%applying CFAR on TS and TSS separately and finally attainin the CD map through performing 'or' operator on two CD maps.

if val_method(1)==1
if value(1)==1
L
mu=L*d/(L-d)
alpha=-2*d*(-2*d + 2*(d^3) + 4*L -5*(d^2)*L - 4*d*(L^2) + 2*(d^3)*(L^2) + 8*(L^3) - 4*(d^2)*(L^3) + 2*d*(L^4))/(2*(d^4) + 12*d*L - 12*(d^3)*L + (d^5)*L - 24*(L^2) + 16*(d^2)*(L^2) - 2*(d^4)*(L^2) - 4*d*(L^3) + (d^3)*(L^3))
lambda=(-6*d^2 + 6*d^4 + 26*d*L - 27*d^3*L + d^5*L - 28*L^2 + 23*d^2*L^2 + d^4*L^2 + 18*d*L^3 - 9*d^3*L^3 - 20*L^4 + 11*d^2*L^4 - 4*d*L^5)/(-2*d^2 + 2*d^4 + 14*d*L - 13*d^3*L + d^5*L - 20*L^2 + 21*d^2*L^2 - 3*d^4*L^2 - 6*d*L^3 + 3*d^3*L^3)

u=1;
CD=127.5*ones(r_m*c_m,1);
for i=1:r_m
    for j=1:c_m
        p=1-pfa;
       t_l=fishercdfinv(p,alpha,lambda,mu);
       %Generation of CD
       if ts(i,j)>=t_l
           CD(u)=255;
       end
       u=u+1;
    end
end
C_D=(reshape(CD,c_m,r_m))';
axes(handles.axes23)
axis off;
imshow(C_D);    
        

    
elseif value(2)==1
sg2=2*sg;
nr_m=r_m+sg2;%new r_m and c_m based on bellow discriptions*
nc_m=c_m+sg2;
ts_c=.1*ones(nr_m,nc_m);%*changes ts:a strip of 4 pixel(in case of gau_s==9) with a value equal to .1 is added around the test statistic image
m=1;

r_ts=reshape(ts',r_m*c_m,1);
for i=1:nr_m
    for j=1:nc_m
        if i<(sg + 1)  ||  j<(sg + 1)  ||  i>(r_m+sg)  ||  j>(c_m+sg)
            continue
        end
        ts_c(i,j)=r_ts(m);
        if ts_c(i,j)<=1
            ts_c(i,j)=1.1;
        end
        m=m+1;
    end
end
u=1;
%mmm=0;
CD=zeros(r_m*c_m,1);
for i=1:nr_m
    for j=1:nc_m
        
        
        if i<(sg + 1)  ||  j<(sg + 1)  ||  i>(r_m+sg)  ||  j>(c_m+sg)
            continue
        end

        
        %towards local threshold calculation..
        
        sum_log_ts=0;%This parameter is equal to summation of log(ts).(class n1) 
        sum_log_ts2=0;%This parameter is equal to (summation of (log(ts)^2) minus nu1^2.
        sum_log_ts3=0;%This parameter is equal to summation of (log(ts)-n1nu1)^3.
        sum_ts=0;
        %sum_ts2=0;
        %sum_ts3=0;
        num_sum=0;
        
        for rr=(i-sg):(i+sg)
            for cc=(j-sg):(j+sg)
                ind=0;
                for l=(i-st):(i+st)
                    for m=(j-st):(j+st)  
                        if rr==l && cc==m
                            ind=1;
                            break
                        end
                    end
                end
                if ind==1
                    continue
                end 
            sum_ts=sum_ts+ts_c(rr,cc);
            %sum_ts2=sum_ts2+ (ts_c(rr,cc).^2);
            %sum_ts3=sum_ts3+ (ts_c(rr,cc).^3);
            sum_log_ts=sum_log_ts+log(ts_c(rr,cc));
            num_sum=num_sum+1;      
            end
        end
        mom1=sum_ts/num_sum;
        k1=sum_log_ts/num_sum;
         
        for rr=(i-sg):(i+sg)
            for cc=(j-sg):(j+sg)
                
                ind=0;
                for l=(i-st):(i+st)
                    for m=(j-st):(j+st)  
                        if rr==l && cc==m
                            ind=1;
                            break
                        end
                    end
                end
                if ind==1
                   continue
                end 
          sum_log_ts2=sum_log_ts2+(log(ts_c(rr,cc))-k1).^2;
          sum_log_ts3=sum_log_ts3+(log(ts_c(rr,cc))-k1).^3;     
            end
        end         
         k2=sum_log_ts2/(num_sum);
         k3=sum_log_ts3/(num_sum);
 
       init = fzero(@(x) (psi(1,x*x) - 0.5*abs(mom1)),1);
       init= abs(init * init);
       p0=zeros(1,2);
       p0(1,1)=50*init;
       p0(1,2)=40*init;

        % p0=[33 10];
         
         
%Non-linear least square method
%        l_o=[k2;k3];
%        maxiter=2;
%        for iter=1:maxiter
%        xx=p0(1,1);
%        yy=p0(2,1);
%        l_c=[(psi(1,xx)+psi(1,yy));(psi(2,xx)-psi(2,yy))];
%        A=[psi(2,xx) psi(2,yy);psi(3,xx) -psi(3,yy)];
%        delta_l=l_o - l_c;
%        delta_x=((A'*A)\A')*delta_l;
%        error=sqrt(delta_x(1)^2+delta_x(2)^2);
%        if error<0.01
%           break
%        end
%        p0=p0+delta_x;
%        end
%        alpha=p0(1);
%        lambda=p0(2);

%nonlinear LQ by lsnonlin function
       options = optimset('Display','off');
       lb =[0;0]; %Vector of lower bounds
       ub=[1000;100000000]; %Vector of upper bounds
       x=lsqnonlin(@logcumCFAR,p0,lb,ub,options);
       alpha=x(1);
       lambda=x(2);
       %mu=mom1;
       mu=(alpha*(exp(k1-psi(alpha)+psi(lambda))))/lambda;

       p=1-pfa;
       t_l=fishercdfinv(p,alpha,lambda,mu);
       %Generation of CD
       %ts_c(i,j)
       %t_l
       if ts_c(i,j)>t_l
           CD(u,1)=255;
           %CD(u,1)
%            mmm=mmm+1
       end
       u=u+1;
    end
end
%save('CD.mat','CD')
C_D=(reshape(CD,c_m,r_m))';
%save('C_D.mat','C_D')
axes(handles.axes23)
axis off;
imshow(C_D); 



%save('T_M.mat','T_M')
rc=0;fp=0;fn=0;tp=0;n_nc=0;n_c=0;
for i=1:r_m
    for j=1:c_m
    if T_M(i,j)==255
        n_c=n_c+1;
        rc=rc+1;
        if C_D(i,j)==255
            tp=tp+1;
        elseif C_D(i,j)==0
            fn=fn+1;
        end
    elseif T_M(i,j)==127.5
        n_nc=n_nc+1;
        if C_D(i,j)==255
            fp=fp+1;
        end
    end
    end
end
daccuracy=(tp/rc)*100;
falarm=(fp/n_nc)*100;
overallerror=((fn+fp)/(n_c+n_nc))*100;
end


LoGT=isempty(find(T_M(:,:)~=0));%lack of ground truth
if LoGT==1
set(handles.edit14,'string','NotAvailable');
set(handles.edit15,'string','NotAvailable');
set(handles.edit16,'string','NotAvailable');
else
set(handles.edit14,'string',daccuracy);
set(handles.edit15,'string',falarm);
set(handles.edit16,'string',overallerror);
end



elseif val_method(2)==1
if value(1)==1

    
    
    
elseif value(2)==1
    
    
    
    
    
    
end    
end
function [x]=fishercdfinv(P,a,b,mu)
x=finv(P,2*a,2*b);
x=x.*mu;
function F=logcumCFAR(x)
global k2 k3
F= [psi(1,x(1))+psi(1,x(2))-k2;psi(2,x(1))-psi(2,x(2))-k3];
function F=mixedPE_KI1(x)
global n1m1 n1w1 n1w2 n1m2 n1w0
F= [-n1m1+x(2)*x(3)/(x(2)-1);-(n1w1/n1m1)+n1w0+(1/x(1))+(1/(x(2)-1)); -(n1w2/n1m2)+(n1w1/n1m1)+(1/(x(1)+1))+(1/(x(2)-2))];
function F=mixedPE_KI2(x)
global n2m1 n2w1 n2w2 n2m2 n2w0
F= [-n2m1+x(2)*x(3)/(x(2)-1);-(n2w1/n2m1)+n2w0+(1/x(1))+(1/(x(2)-1)); -(n2w2/n2m2)+(n2w1/n2m1)+(1/(x(1)+1))+(1/(x(2)-2))];
function F=logcum_KI1(x)
global n1nu2 n1nu3
F= [psi(1,x(1))+psi(1,x(2))-n1nu2;psi(2,x(1))-psi(2,x(2))-n1nu3];
function F=logcum_KI2(x)
global n2nu2 n2nu3
F= [psi(1,x(1))+psi(1,x(2))-n2nu2;psi(2,x(1))-psi(2,x(2))-n2nu3];
function F=n1kGGamma(x)
global n1nu2 n1nu3;
F=(n1nu2^3/n1nu3^2)-(psi(1,x)^3)/(psi(2,x)^2);
function F=n2kGGamma(x)
global n2nu2 n2nu3;
F=(n2nu2^3/n2nu3^2)-(psi(1,x)^3)/(psi(2,x)^2);
function F=n1kGamma(x)
global n1nu2;
F=psi(1,x)-n1nu2;
function F=n2kGamma(x)
global n2nu2;
F=psi(1,x)-n2nu2;


% --------------------------------------------------------------------
function Untitled_13_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.uipanel4,'visible','off');
set(handles.uipanel6,'visible','off');
set(handles.uipanel15,'visible','off');
set(handles.uipanel24,'visible','on');
set(handles.uipanel35,'visible','off');
set(handles.uipanel37,'visible','off');
set(handles.uipanel44,'visible','off');
function edit14_Callback(hObject, eventdata, handles)
% hObject    handle to edit14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit14 as text
%        str2double(get(hObject,'String')) returns contents of edit14 as a double


% --- Executes during object creation, after setting all properties.
function edit14_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit15_Callback(hObject, eventdata, handles)
% hObject    handle to edit15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit15 as text
%        str2double(get(hObject,'String')) returns contents of edit15 as a double


% --- Executes during object creation, after setting all properties.
function edit15_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit16_Callback(hObject, eventdata, handles)
% hObject    handle to edit16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit16 as text
%        str2double(get(hObject,'String')) returns contents of edit16 as a double


% --- Executes during object creation, after setting all properties.
function edit16_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton12.
function pushbutton12_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global t_s d ts L r r_m c_m T_M TS TSS val_type
r_m
c_m
% val1=get(handles.radiobutton45,'value');%real dataset(suzhou)
% val2=get(handles.radiobutton48,'value');%simulated dataset
maxts=255;
E=1.e10*ones(((maxts-.1)/.02)+1,1);
taw_vals=zeros(((maxts-.1)/.02)+1,1);
ind=1;
for taw=0.1:0.02:maxts
 %C_D=zeros(r_m,c_m);
C_D=127.50*ones(r_m,c_m);
for x=1:r_m
    for y=1:c_m
        if (ts(x,y))>(taw-.02)
        C_D(x,y)=255;
        end
    end
end

fp=0;fn=0;
for i=1:r_m
    for j=1:c_m
    if T_M(i,j)==255
        if C_D(i,j)==127.5
            fn=fn+1;
        end
    elseif T_M(i,j)==127.5
        if C_D(i,j)==255
            fp=fp+1;
        end
    end
    end
end
E(ind)=fn+fp;
taw_vals(ind)=taw;
ind=ind+1;
end
%E(1:10,1)
[min_E,min_error_taw_ind]=min(E)
min_errorrate_taw=taw_vals(min_error_taw_ind)
C_D=127.50*ones(r_m,c_m);
for x=1:r_m
    for y=1:c_m
        if (ts(x,y))>(min_errorrate_taw-.02)
        C_D(x,y)=255;
        end
    end
end
C_D1=uint8(C_D);
 axes(handles.axes24)
 axis off;
 imshow(C_D1);
 figure(22)
 clf
 imshow(C_D1)
%numerical evaluation

rc=0;fp=0;fn=0;tp=0;n_nc=0;

for i=1:r_m
    for j=1:c_m
    if T_M(i,j)==255
        rc=rc+1;
        if C_D(i,j)==255
            tp=tp+1;
        elseif C_D(i,j)==127.5
            fn=fn+1;
        end
    elseif T_M(i,j)==127.5
        n_nc=n_nc+1;
        if C_D(i,j)==255
            fp=fp+1;
        end
    end
    end
end


rc
n_nc
fn
fp
daccuracy=(tp/rc)*100;
falarm=(fp/n_nc)*100;
overallerror=((fn+fp)/(rc+n_nc))*100; 

LoGT=isempty(find(T_M(:,:)~=0));%lack of ground truth
if LoGT==1
set(handles.edit17,'string','NotAvailable');
set(handles.edit18,'string','NotAvailable');
set(handles.edit19,'string','NotAvailable');
else
set(handles.edit17,'string',daccuracy);
set(handles.edit18,'string',falarm);
set(handles.edit19,'string',overallerror);
end

function edit17_Callback(hObject, eventdata, handles)
% hObject    handle to edit17 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit17 as text
%        str2double(get(hObject,'String')) returns contents of edit17 as a double


% --- Executes during object creation, after setting all properties.
function edit17_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit17 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit18_Callback(hObject, eventdata, handles)
% hObject    handle to edit18 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit18 as text
%        str2double(get(hObject,'String')) returns contents of edit18 as a double


% --- Executes during object creation, after setting all properties.
function edit18_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit18 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit19_Callback(hObject, eventdata, handles)
% hObject    handle to edit19 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit19 as text
%        str2double(get(hObject,'String')) returns contents of edit19 as a double


% --- Executes during object creation, after setting all properties.
function edit19_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit19 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton15.
function pushbutton15_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global t_s d ts L r r_m c_m T_M TS TSS t_ss1 t_s1 srw
format long
val_method(1,1)=get(handles.radiobutton55,'value');
val_method(2,1)=get(handles.radiobutton56,'value') ;
if val_method(1)==1

%histogram of quantized trace image
ts=TS;
maxts=max(max(ts));
%max(max(ts));
h1=zeros(maxts+1,1);
kk=zeros(maxts+1,1);
for k=0:maxts
    kk(k+1)=k;
    for l=1:r_m
        for m=1:c_m
            if k==0
                kk(k+1)=.1;
                if (ts(l,m))==.1 && T_M(l,m)~=0
                    h1(k+1)=h1(k+1)+1;
                end
            end
            if (ts(l,m))==k && T_M(l,m)~=0
                h1(k+1)=h1(k+1)+1;
            end
        end
    end
end
%h=h1/(r_m*c_m);
N=size(find(T_M~=0),1);
%N=r_m*c_m;
h=h1/N;
axes(handles.axes29);
 %k_pr=10*log10(kk);
% h_pr=10*log10(h);
bar(kk,h1,'b');
 
 figure(1);
 clf
 linenu = 1.5;fs=20;
 %[fA,xA] = [h,kk];
% [fB,xB] = ksdensity(Mj2,'npoints',10000);
% [fC,xC] = ksdensity(Mj5,'npoints',10000);
 xvector=kk';
 fvector=h1';
 plot(xvector, fvector, 'LineWidth',linenu);
 legend({'Histogram of TS image'},'FontSize',fs)
 xlabel('Gray level', 'fontsize',fs)
 ylabel('Frequency', 'fontsize',fs)
set(gca,'FontSize',fs);

E=-1.e10*ones(maxts+1,1);
taw_vales=zeros(maxts+1,1);
ind=1;
%(maxts-(maxts/3))
for taw=0:maxts
    if h(taw+1)==0
        taw_vales(ind)=taw;
        ind=ind+1;
        continue
    else
       
            P_B=0;
            H_A=0;       
            for q=0:taw
                P_B=P_B+h(q+1);
                H_A=H_A+h(q+1).*log(h(q+1)/sum(h(1:(taw+1),1)))./sum(h(1:(taw+1),1));
                
            end

            H_B=0;       
            for w=taw+1:maxts
                H_B=H_B+h(q+1).*log(h(q+1)/(1-P_B))./(1-P_B);      
            end
            taw_vales(ind)=taw;
            E(ind)=-(H_A+H_B);
            ind=ind+1;
    end
end

[max_E,opt_taw_ind]=max(E);
opt_taw=taw_vales(opt_taw_ind);
%change or no-change map 

%C_D:change detection image (white:change, black:no change)

%C_D=zeros(r_m,c_m);
C_D=127.50*ones(r_m,c_m);
for x=1:r_m
    for y=1:c_m
        if (ts(x,y))>(opt_taw+1)
        C_D(x,y)=255;
        end
    end
end
C_D=uint8(C_D);
axes(handles.axes28)
axis off;
imshow(C_D);

elseif val_method(2)==1

end
% --------------------------------------------------------------------
function Untitled_15_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.uipanel4,'visible','off');
set(handles.uipanel6,'visible','off');
set(handles.uipanel15,'visible','off');
set(handles.uipanel24,'visible','off');
set(handles.uipanel35,'visible','off');
set(handles.uipanel37,'visible','on');
set(handles.uipanel44,'visible','off');

% --------------------------------------------------------------------
function Untitled_14_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.uipanel4,'visible','off');
set(handles.uipanel6,'visible','off');
set(handles.uipanel15,'visible','off');
set(handles.uipanel24,'visible','off');
set(handles.uipanel35,'visible','on');
set(handles.uipanel37,'visible','off');
set(handles.uipanel44,'visible','off');

% --- Executes on button press in pushbutton17.
function pushbutton17_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton17 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global t_s d ts L r r_m c_m T_M TS TSS t_ss1 t_s1 srw val_type
format long
val_method(1,1)=get(handles.radiobutton61,'value');
val_method(2,1)=get(handles.radiobutton62,'value') ;
if val_method(1)==1

%histogram of quantized trace image
%ts=TSS;
maxts=max(max(ts));
%(maxts-255)/maxts;
for i=1:r_m
    for j=1:c_m
         if ts(i,j)==.1
             ts(i,j)=0;
         end
        %ts(i,j)=((255-maxts)*ts(i,j)/maxts+maxts)*ts(i,j)/maxts;
        ts(i,j)=255*ts(i,j)/maxts;
         ts(i,j)=fix(ts(i,j));
    end
end
%((255-maxts)*1000/maxts+maxts)*1000/maxts
 %maxts=22;

maxts=max(max(ts));
h1=zeros(maxts+1,1);
kk=zeros(maxts+1,1);
for k=0:maxts
    kk(k+1)=k;
    for l=1:r_m
        for m=1:c_m
            if k==0
                kk(k+1)=0;
                if (ts(l,m))==0 && T_M(l,m)~=0 %|| (l<50 && m>250)
                   if m<145
                    h1(k+1)=h1(k+1)+1;
                   end
                end
            end
            if (ts(l,m))==k && T_M(l,m)~=0 %|| (l<50 && m>250)
               if m<145
                h1(k+1)=h1(k+1)+1;
               end
            end
        end
    end
end
h=h1/200;
%N=size(find(T_M~=0),1);
% N=r_m*c_m;
% h=h1/N;
axes(handles.axes32);
 %k_pr=10*log10(kk);
% h_pr=10*log10(h);
bar(kk,h,'b');
%  
% 
%histogram of ts plot
%  figure(1);
%  clf
%  linenu = 1.5;fs=13;
%  %[fA,xA] = [h,kk];
% % [fB,xB] = ksdensity(Mj2,'npoints',10000);
% % [fC,xC] = ksdensity(Mj5,'npoints',10000);
%  xvector=kk';
%  h1=h1/(r_m*c_m);
%  fvector=h1';plot(xvector, fvector, 'LineWidth',linenu);
%hold on
%  aa=[1 1];
%  bb=[0 .27];
%  plot(aa,bb,'r')
%  hold on
%  aaa=[10 10];
%  bbb=[0 .27];
%  plot(aaa,bbb,'g')
%  %legend({'Histogram of TS image'},'FontSize',fs)
%  xlabel('Gray level', 'fontsize',fs)
%  ylabel('Frequency', 'fontsize',fs)
% set(gca,'FontSize',fs);

E=-1.e10*ones(maxts+1,1);
taw_vales=zeros(maxts+1,1);
ind=1;
m_T=sum(h(1:maxts+1,1));%mu_total
sig_T2=0;
for num=0:maxts
    sig_T2=sig_T2+h(num+1).*(num-m_T)^2;%sigma_T.^2
end

for taw=0:255
            w_t=0;%omega_taw
            m_t=0;%mu_taw       
            for q=0:taw
                w_t=w_t+h(q+1);
                m_t=m_t+q.*h(q+1);
                
            end
            m_t2=(m_T-w_t*m_t)/(1-w_t);
            taw_vales(ind)=taw;
            mu_0=m_t/w_t;mu_1=(m_T-m_t)./(1-w_t);
            sig0=0;sig1=0;
            for q=0:taw
                sig0=sig0+h(q+1).*(q-mu_0)^2;    
            end 
            sig_02=sig0/w_t;
            for q=taw+1:maxts
                sig1=sig1+h(q+1).*(q-mu_1)^2;    
            end            
            sig_12=sig1/(1-w_t);
            sig_w2=w_t*sig_02+(1-w_t)*sig_12;%within class variance
            sig_b2=w_t*(1-w_t)*(mu_1-mu_0)^2;%between class variance    
            new_criterion=(1-h(taw+1))*(w_t*(m_t^2))+((1-w_t)*(m_t2^2));
            E(ind)=new_criterion;%sig_b2;%new_criterion;%sig_b2;%new_criterion;%sig_b2;%new_criterion;%sig_b2;%sig_b2;%sig_b2;%sig_T2/sig_w2;%sig_b2/sig_T2;%new_criterion;%sig_b2/sig_T2;%sig_T2/sig_w2;
            ind=ind+1;
    
end




 
%  figure(2);
%  clf
%  linenu = 1.5;fs=20;
%  %[fA,xA] = [h,kk];
% % [fB,xB] = ksdensity(Mj2,'npoints',10000);
% % [fC,xC] = ksdensity(Mj5,'npoints',10000);
%  plot(E(1:100),'*b', 'LineWidth',linenu);
%  hold on
%  plot(E(1:100), 'LineWidth',linenu)
% % legend({'Histogram of TS image'},'FontSize',fs)
%  xlabel('Threshold', 'fontsize',fs)
%  ylabel('Criterion Function', 'fontsize',fs)
% set(gca,'FontSize',fs);




%save('E_r.mat','E')
[max_E,opt_taw_ind]=max(E);
opt_taw=taw_vales(opt_taw_ind)
%change or no-change map 
% E1=E;
% save('E_l.mat','E1')
%C_D:change detection image (white:change, black:no change)

%C_D=zeros(r_m,c_m);
C_D=127.50*ones(r_m,c_m);
for x=1:r_m
    for y=1:c_m
        if (ts(x,y))>(opt_taw)
        C_D(x,y)=255;
        end
    end
end
C_D1=uint8(C_D);
axes(handles.axes31)
axis off;
imshow(C_D1);
% imwrite(C_D1,'sh7_2_imwrite.jpg')
figure(1)
clf
imshow(C_D1);
axis off
% indd=1;
% for opt_taw=0:255
% C_D=127.50*ones(r_m,c_m);
% for x=1:r_m
%     for y=1:c_m
%         if (ts(x,y))>(opt_taw)
%         C_D(x,y)=255;
%         end
%     end
% end
% %numerical evaluation
% rc=0;fp=0;fn=0;tp=0;n_nc=0;
% for i=1:r_m
%     for j=1:c_m
%     if T_M(i,j)==255
%         rc=rc+1;
%         if C_D(i,j)==255
%             tp=tp+1;
%         elseif C_D(i,j)==127.5
%             fn=fn+1;
%         end
%     elseif T_M(i,j)==0
%         n_nc=n_nc+1;
%         if C_D(i,j)==255
%             fp=fp+1;
%         end
%     end
%     end
% end
% overallerror(indd)=((fn+fp)/(n_nc+rc))*100;
% indd=indd+1;
% end
% save('oerror.mat','overallerror')






%numerical evaluation
rc=0;fp=0;fn=0;tp=0;n_nc=0;
for i=1:r_m
    for j=1:c_m
    if T_M(i,j)==255
        rc=rc+1;
        if C_D(i,j)==255
            tp=tp+1;
        elseif C_D(i,j)==127.5
            fn=fn+1;
        end
    elseif T_M(i,j)==127.5
        n_nc=n_nc+1;
        if C_D(i,j)==255
            fp=fp+1;
        end
    end
    end
end


daccuracy=(tp/rc)*100;
falarm=(fp/n_nc)*100;
overallerror=((fn+fp)/(n_nc+rc))*100;


if val_type(2)==1%ground truth corresponding to real data
    
rc=0;fp=0;fn=0;tp=0;n_nc=0;n_c=0;
for i=1:r_m
    for j=1:c_m
    if T_M(i,j)==255
        n_c=n_c+1;
        rc=rc+1;
        if C_D(i,j)==255
            tp=tp+1;
        elseif C_D(i,j)==127.5
            fn=fn+1;
        end
    elseif T_M(i,j)==127.5
        n_nc=n_nc+1;
        if C_D(i,j)==255
            fp=fp+1;
        end
    end
    end
end
daccuracy=(tp/rc)*100;
falarm=(fp/n_nc)*100;
overallerror=((fn+fp)/(n_c+n_nc))*100;
end


LoGT=isempty(find(T_M(:,:)~=0));%lack of ground truth
if LoGT==1
set(handles.edit29,'string','NotAvailable');
set(handles.edit30,'string','NotAvailable');
set(handles.edit31,'string','NotAvailable');
else
set(handles.edit29,'string',daccuracy);
set(handles.edit30,'string',falarm);
set(handles.edit31,'string',overallerror);
end

elseif val_method(2)==1

end




function edit23_Callback(hObject, eventdata, handles)
% hObject    handle to edit23 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit23 as text
%        str2double(get(hObject,'String')) returns contents of edit23 as a double


% --- Executes during object creation, after setting all properties.
function edit23_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit23 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton16.
function pushbutton16_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit24_Callback(hObject, eventdata, handles)
% hObject    handle to edit24 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit24 as text
%        str2double(get(hObject,'String')) returns contents of edit24 as a double


% --- Executes during object creation, after setting all properties.
function edit24_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit24 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit25_Callback(hObject, eventdata, handles)
% hObject    handle to edit25 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit25 as text
%        str2double(get(hObject,'String')) returns contents of edit25 as a double


% --- Executes during object creation, after setting all properties.
function edit25_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit25 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit26_Callback(hObject, eventdata, handles)
% hObject    handle to edit26 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit26 as text
%        str2double(get(hObject,'String')) returns contents of edit26 as a double


% --- Executes during object creation, after setting all properties.
function edit26_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit26 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit27_Callback(hObject, eventdata, handles)
% hObject    handle to edit27 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit27 as text
%        str2double(get(hObject,'String')) returns contents of edit27 as a double


% --- Executes during object creation, after setting all properties.
function edit27_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit27 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit28_Callback(hObject, eventdata, handles)
% hObject    handle to edit28 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit28 as text
%        str2double(get(hObject,'String')) returns contents of edit28 as a double


% --- Executes during object creation, after setting all properties.
function edit28_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit28 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------
function Untitled_16_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.uipanel4,'visible','off');
set(handles.uipanel6,'visible','off');
set(handles.uipanel15,'visible','off');
set(handles.uipanel24,'visible','off');
set(handles.uipanel35,'visible','off');
set(handles.uipanel37,'visible','off');
set(handles.uipanel44,'visible','on');



function edit29_Callback(hObject, eventdata, handles)
% hObject    handle to edit29 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit29 as text
%        str2double(get(hObject,'String')) returns contents of edit29 as a double


% --- Executes during object creation, after setting all properties.
function edit29_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit29 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit30_Callback(hObject, eventdata, handles)
% hObject    handle to edit30 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit30 as text
%        str2double(get(hObject,'String')) returns contents of edit30 as a double


% --- Executes during object creation, after setting all properties.
function edit30_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit30 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit31_Callback(hObject, eventdata, handles)
% hObject    handle to edit31 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit31 as text
%        str2double(get(hObject,'String')) returns contents of edit31 as a double


% --- Executes during object creation, after setting all properties.
function edit31_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit31 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in radiobutton68.
function radiobutton68_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton68 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton68


% --- Executes on button press in radiobutton67.
function radiobutton67_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton67 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton67


% --- Executes on button press in radiobutton66.
function radiobutton66_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton66 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton66


% --- Executes on button press in radiobutton65.
function radiobutton65_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton65 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton65


% --- Executes on button press in radiobutton64.
function radiobutton64_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton64 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton64


% --- Executes on button press in radiobutton63.
function radiobutton63_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton63 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton63


% --- Executes on button press in pushbutton18.
function pushbutton18_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton18 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit34_Callback(hObject, eventdata, handles)
% hObject    handle to edit34 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit34 as text
%        str2double(get(hObject,'String')) returns contents of edit34 as a double


% --- Executes during object creation, after setting all properties.
function edit34_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit34 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit35_Callback(hObject, eventdata, handles)
% hObject    handle to edit35 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit35 as text
%        str2double(get(hObject,'String')) returns contents of edit35 as a double


% --- Executes during object creation, after setting all properties.
function edit35_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit35 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit36_Callback(hObject, eventdata, handles)
% hObject    handle to edit36 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit36 as text
%        str2double(get(hObject,'String')) returns contents of edit36 as a double


% --- Executes during object creation, after setting all properties.
function edit36_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit36 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
