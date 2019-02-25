% h2v: to transform from hidden (output) variables to visible (input) variables
%
% V = h2v(dnn, H)
%
%
%Output parameters:
% V: visible (input) variables, where # of row is number of data and # of col is # of visible (input) nodes
%
%
%Input parameters:
% dnn: the Deep Neural Network model (dbn, rbm)
% H: hidden (output) variables, where # of row is number of data and # of col is # of hidden (output) nodes
%
%
%Example:
% datanum = 1024;
% outputnum = 16;
% inputnum = 4;
%
% outputdata = rand(datanum, outputnum);
%
% dnn = randRBM( inputnum, outputnum );
% inputdata = h2v( dnn, outputdata );
%
%
%Version: 20130830

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:                                     %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function V = h2v(dnn, H)

ind1 = numel(dnn.type);
ind0 = ind1-2;
type = dnn.type(ind0:ind1);

if( isequal(dnn.type, 'BBRBM') )
    V = sigmoid( bsxfun(@plus, H * dnn.W', dnn.visbias ) );

elseif(isequal(dnn.type, 'mix'))
    v_b = sigmoid( bsxfun(@plus, H * dnn.W(dnn.dimV_con+1:end,:)', dnn.visbias(dnn.dimV_con+1:end) ) );
    v_con=bsxfun(@plus,H * dnn.W(1:dnn.dimV_con,:)',dnn.visbias(1:dnn.dimV_con));
    v_con=v_con.*dnn.sig;
    V=[v_con,v_b];
    %v(1:dnn.dimV_con)=bsxfun(@times,v(1:dnn.dimV_con),dnn.sig);
    %V = bsxfun(@plus, v, dnn.visbias );
    
elseif( isequal(dnn.type, 'GBRBM') )
    h = bsxfun(@times, H * dnn.W', dnn.sig);
    V = bsxfun(@plus, h, dnn.visbias );
    
elseif( isequal(dnn.type, 'BBPRBM') )
    w2 = dnn.W .* dnn.W;
    pp = H .* ( 1-H );
    mu = bsxfun(@plus, H * dnn.W', dnn.visbias );
    s2 = pp * w2';
    V = sigmoid( mu ./ sqrt( 1 + s2 * pi / 8 ) );
    
elseif( isequal(type, 'DBN') )
    nrbm = numel( dnn.rbm );
    V0 = H;
    for i=nrbm:-1:1
        V1 = h2v( dnn.rbm{i}, V0 );
        V0 = V1;
    end
    V = V1;
    
end
