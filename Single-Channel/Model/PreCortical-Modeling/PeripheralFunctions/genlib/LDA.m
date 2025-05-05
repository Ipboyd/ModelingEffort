% LDA - MATLAB subroutine to perform linear discriminant analysis
% by Will Dwinnell and Deniz Sevis
%
% Use:
% W = LDA(Input,Target,Priors)
%
% W       = discovered linear coefficients (first row is the constants)
% Input   = predictor data (variables in columns, observations in rows)
% Target  = target variable (class labels)
% Priors  = vector of prior probabilities (optional)
%
% Note: discriminant coefficients are stored in W in the order of unique(Target)
%
% Example:
%
% % Generate example data: 2 groups, of 10 and 15, respectively
% X = [randn(10,2); randn(15,2) + 1.5];  Y = [zeros(10,1); ones(15,1)];
%
% % Calculate linear discriminant coefficients
% W = LDA(X,Y);
%
% % Calulcate linear scores for training data
% L = [ones(25,1) X] * W;
%
% % Calculate class probabilities
% P = exp(L) ./ repmat(sum(exp(L),2),[1 2]);
%
%
% Last modified: Dec-11-2010


function W = LDA(Input,Target,Priors,Reg)

if(nargin < 4 || isempty(Reg))
	Reg = 0;
end
nReg = numel(Reg);

% Determine size of input data
[n m] = size(Input);

% Discover and count unique class labels
ClassLabel = unique(Target);
k = length(ClassLabel);

% Initialize
nGroup     = NaN(k,1);     % Group counts
GroupMean  = NaN(m,k);     % Group sample means
PooledCov  = zeros(m,m);   % Pooled covariance

% Loop over classes to perform intermediate calculations
for i = 1:k,
    % Establish location and size of each class
    Group      = (Target == ClassLabel(i));
    nGroup(i)  = sum(double(Group));
	Temp = Input(Group,:);
    
    % Calculate group mean vectors
    GroupMean(:,i) = mean(Temp);
    
    % Accumulate pooled covariance information
    PooledCov = PooledCov + ((nGroup(i) - 1) / (n - k) ).* cov(Temp);
end

% Assign prior probabilities
if  (nargin >= 3 && ~isempty(Priors))
    % Use the user-supplied priors
    PriorProb = Priors;
else
    % Use the sample probabilities
    PriorProb = nGroup / n;
end

% Loop over classes to calculate linear discriminant coefficients
W = zeros([size(PooledCov,1)+1 2 nReg]);
for ri = 1:nReg
	PooledCov2 = PooledCov*(1-Reg(ri))+(Reg(ri)/m)*trace(PooledCov)*eye(m);
	Temp = PooledCov2 \ GroupMean;
	W(:,:,ri) = [diag(-0.5 * GroupMean.' * Temp + log(PriorProb(i))).'; Temp];
end
% EOF
