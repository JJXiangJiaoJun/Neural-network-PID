function [sys,x0,str,ts,simStateCompliance] = nnbp(t,x,u,flag,T,nh,xite,alfa)
switch flag,
  case 0,
    [sys,x0,str,ts,simStateCompliance]=mdlInitializeSizes(T,nh);
%初始化函数
  case 3,
    sys=mdlOutputs(t,x,u,nh,xite,alfa);
%输出函数
  case {1,2,4,9},
    sys=[];
  otherwise
    DAStudio.error('Simulink:blocks:unhandledFlag', num2str(flag));
end
function [sys,x0,str,ts,simStateCompliance]=mdlInitializeSizes(T,nh)
%调用初始画函数，两个外部输入参数 参数T确定采样时间，参数nh确定隐含层层数
sizes = simsizes;
sizes.NumContStates  = 0;
sizes.NumDiscStates  = 0;
sizes.NumOutputs     = 4+6*nh;
%定义输出变量，包括控制变量u,隐含层+输出层所有加权系数
sizes.NumInputs      = 7+12*nh;
%定义输入变量，包括前7个参数[e(k);e(k-1);e(k-2);y(k);y(k-1);r(k);u(k-1)]
%隐含层+输出层权值系数（k-2),隐含层+输出层权值系数（k-1）
sizes.DirFeedthrough = 1;
sizes.NumSampleTimes = 1; 
sys = simsizes(sizes);
x0  = [];
str = [];
ts  = [T 0];
simStateCompliance = 'UnknownSimState';
function sys=mdlOutputs(t,x,u,nh,xite,alfa)
%调用输出函数
wi_2 = reshape(u(8:7+3*nh),nh,3);
%隐含层（k-2)权值系数矩阵，维数nh*3
wo_2 = reshape(u(8+3*nh:7+6*nh),3,nh);
%输出层（k-2）权值系数矩阵，维数3*nh
wi_1 = reshape(u(8+6*nh:7+9*nh),nh,3);
%隐含层（k-1)权值系数矩阵，维数nh*3
wo_1 = reshape(u(8+9*nh:7+12*nh),3,nh);
%输出层（k-1）权值系数矩阵，维数3*nh
xi = [u(6),u(4),u(1)];
%神经网络的输入xi=[u(6),u(4),u(1)]=[r(k),y(k),e(k)]
xx = [u(1)-u(2);u(1);u(1)+u(3)-2*u(2)];
%xx=[u(1)-u(2);u(1);u(1)+u(3)-2*u(2)]=[e(k)-e(k-1);e(k);e(k)+e(k-2)-2*e(k-1)]
I = xi*wi_1';
%计算隐含层的输入，I=神经网络的输入*隐含层权值系数矩阵的转置wi_1'，结果为：
%I=[net0(k),net1(k)...netnh(k)]为1*nh矩阵
Oh = exp(I)./(exp(I)+exp(-I));
%激活函数，可更改
%计算隐含层的输出，(exp(I)-exp(-I))./(exp(I)+exp(-I))为隐含层的激活函数Sigmoid
%Oh=[o0(k),o1(k)...onh(k)],为1*nh的矩阵
O = wo_1*Oh';
%计算输出层的输入，维数3*1
K = 2./(exp(O)+exp(-O)).^2;
%激活函数，可更改
%计算输出层的输出K=[Kp,Ki,Kd]，维数为1*3
%exp(Oh)./(exp(Oh)+exp(-Oh))为输出层的激活函数Sigmoid
uu = u(7)+K'*xx;
%根据增量式PID控制算法计算控制变量u(k)
dyu = sign((u(4)-u(5))/(uu-u(7)+0.0000001));
%计算输出层加权系数修正公式的sgn
%sign((y(k)-y(k-1))/(u(k)-u(k-1)+0.0000001)近似代表偏导
dK = 2./(exp(K)+exp(-K)).^2;
%激活函数，可更改
delta3 = u(1)*dyu*xx.*dK;
wo = wo_1+xite*delta3*Oh+alfa*(wo_1-wo_2);
%输出层加权系数矩阵的修正
dOh = 2./(exp(Oh)+exp(-Oh)).^2;
%激活函数，可更改
wi = wi_1+xite*(dOh.*(delta3'*wo))'*xi+alfa*(wi_1-wi_2);
%隐含层加权系数修正
sys = [uu;K(:);wi(:);wo(:)];
%输出层输出sys=[uu;K(:);wi(:);wo(:)]=
%[uu;Kp;Ki;Kd;隐含层+输出层所有权值系数]
%K(:),wi(:),wo(:),把这三个矩阵按顺序排为列向量
