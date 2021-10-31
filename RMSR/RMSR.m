%Main code of "Relation-Induced Multi-Modal Shared Representation Learning
%for Alzheimer's Disease Diagnosis"

function [P,W]=RMSRL(X,Y,k,itenum,gamma1,gamma2,gamma3,lambda1,lambda2,lambda3,lambda4,rho1,rho2,rho3,rho4)
[n,m]=size(X);
Im=eye(m);
Ik=eye(k);
Q=randn(k,m);
P=randn(m,k);
W=randn(k,1);
U=randn(n,k);

U_update=zeros(n,k);
[L,S]=mygraph(Y);
for t=1:itenum
    %step1 update U
    delta_X1=2*gamma1*(U*W*W'-Y*W');
    delta_X2=2*gamma2*(U-X*P);
    delta_X3=2*gamma3*(U*Q*Q'-X*Q');
    for temp=1:k
        U_update(:,temp)=decol(U,temp);
    end
    delta_X4=lambda1*U_update;
    delta_X5=-2*lambda2*Y*Y'*U;
    delta_X6=2*lambda3*L*U;
    delta_X7=-2*lambda4*S*U;
    delta_X8=2*rho1*U;
    delta_X=delta_X1+delta_X2+delta_X3+delta_X4+delta_X5+delta_X6+delta_X7+delta_X8;
    U=U-delta_X;
    U=normalize(U);
    
    %step2 update P
    temp1=gamma2*(X'*X)+rho2*Im;
    temp1(temp1==Inf)=eps; temp1(find(isnan(temp1)))=eps;
    P=temp1\(gamma2*(X'*U));
    
    %step3 update Q
    temp2=gamma3*(U'*U)+rho3*Ik;
    temp2(temp2==Inf)=eps; temp2(find(isnan(temp2)))=eps;
    Q=temp2\(gamma3*(U'*X));
    
    %step4 update W
    temp3=gamma1*(U'*U)+rho4*Ik;
    temp3(temp3==Inf)=eps; temp3(find(isnan(temp3)))=eps;
    W=temp3\(gamma1*(U'*Y));
    
    J(t)=gamma1*norm(Y-U*W,'fro')^2+gamma2*norm(U-X*P,'fro')^2+gamma3*norm(X-U*Q,'fro')^2+...
        lambda1*trace(U'*U_update)-lambda2*trace(U'*Y*Y'*U)+lambda3*trace(U'*L*U)-lambda4*trace(U'*S*U)+...
        rho1*norm(U,'fro')^2+rho2*norm(P,'fro')^2+rho3*norm(Q,'fro')^2+rho4*norm(W,'fro')^2;
    
end
end


function U_rest=decol(U,i)
[~,m]=size(U);
U_noi=U;
U_noi(:,i)=[];
for temp0=1:(m-1)
    if U(:,i)'*U_noi(:,temp0)<0
        U_noi(:,temp0)=-U_noi(:,temp0);
    end
end
U_rest=sum(U_noi,2);
end

function [L,S]=mygraph(label)
labelkind=unique(label);
labelnum=length(labelkind);
n=size(label,1);
G=zeros(n);
for index=1:labelnum
    classidx=find(label==labelkind(index));
    classnum=sum(label==labelkind(index));
    for k=1:classnum
        for j=1:classnum
            G(classidx(k),classidx(j))=1/classnum;
        end
    end
end
S=sparse(G);
L=eye(n)-S;
end
