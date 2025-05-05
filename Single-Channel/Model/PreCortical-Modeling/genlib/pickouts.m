function varargout = pickouts(outNums, funHandle, varargin)

outs = cell(1,max(outNums));
[outs{:}] = funHandle(varargin{:});
varargout = outs(outNums);