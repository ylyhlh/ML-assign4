--[[
Mixture of Gaussians Implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 11/28/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

The mixture of gaussians algorithm should be presented here. You can implement
it in anyway you want. For your convenience, a multivariate gaussian object is
provided at gaussian.lua.

Here is how I implemented it:

mog(n,k) is a constructor to return an object m which will perform MoG
algorithm on data of dimension n with k gaussians. The m object stores the i-th
gaussian at m[i], which is a gaussian object. The m object has the following
methods:

m:g(x): The decision function which returns a vector of k elements indicating
each gaussian's likelihood

m:f(x): The output function to output a prototype that could replace vector x.

m:learn(x,p,eps): learn the gaussians using x, which is an m*n matrix
representing m data samples. p is regularization to keep each gaussian's
covariance matrices non-singular. eps is a stop criterion.
]]

dofile("gaussian.lua")
dofile("kmeans.lua")

-- Create a MoG learner
-- n: dimension of data
-- k: number of gaussians
function mog(n,k)
-- Remove the following line and add your stuff
--print("You have to define this function by yourself!");
   local m={}
   m.features=n--number of features
   m.gaussianSize=k--number of gaussians
   m.datasize=0--number of titles
   m.respons=torch.Tensor(1,1):fill(0)--responsibilities
   m.M=torch.Tensor(m.gaussianSize,m.features)--means
   m.W=torch.Tensor(m.gaussianSize)--mixing coefficients

   --creat the gaussians
   for i=1,m.gaussianSize do
      m[i]=gaussian(n)
   end

   
   --m:g(x): The decision function which returns a vector of k elements indicating each gaussian's likelihood
   function m:g(x)
     local weight=torch.Tensor(m.gaussianSize):fill(0)
     for j=1,m.gaussianSize do
        weight[j]=m[j]:eval(x)*m.W[j]
     end
     local sum=torch.sum(weight)
     weight:div(sum)
     return weight
   end
   --m:f(x): The output function to output a prototype that could replace vector x.
   function m:f(x)
      local max,index=torch.max(m:g(x),1)
      return m[index[1]].m
   end
   
   --m:learn(x,p,eps): learn the gaussians using x, which is an m*n matrix representing m data samples. p is regularization to keep each gaussian's covariance matrices non-singular. eps is a stop criterion.
   function m:learn(x,p,eps)
      --init the gaussian with kmeans
      m.datasize=x:size(1)
      local mk=kmeans(m.features,m.gaussianSize)
      mk:learn(x)
      --init the responsibility with binary result from k means
      m.respons:resize(m.datasize,m.gaussianSize):fill(0)
      --for i=1,m.datasize do
      --   m.respons[i][mk.center[i]]=1
      --end
      m.respons:copy(mk.respons)
      --first is M step to get W, mean M, covariances stored in gaussians
      m:mstep(x,p)
      --m:estep(x)
      err0=m:evalue(x)
      for i=1,10 do
          print(i)
         m:estep(x)
         m:mstep(x,p)
         err1=m:evalue(x)
         print(torch.abs((err0-err1)/err0))
         if torch.abs((err0-err1)/err0)<eps then 
            break
         end
         --if loss become bigger
         if (err0-err1)/err0>10*eps then 
            break
         end
         err0=err1
      end
      --begin the iterations
   end

   
   --M step
   function m:mstep(x,p)
      local Nk=torch.sum( m.respons, 1)[1]
      --print(m.respons:select(2,1))
      for i=1,m.gaussianSize do
         m[i]:learn(x,m.respons:select(2,i),p)
      end
      m.W=torch.div(Nk,m.datasize)
   end
   --E step
   function m:estep(x)
      for i=1,m.datasize do
         local weight=m:g(x[i])
         m.respons:select(1,i):copy(weight)
      end
   end

   function m:evalue(x)
      local err=0
      for i=1,m.datasize do
         local weight=torch.Tensor(m.gaussianSize):fill(0)
         for j=1,m.gaussianSize do
            weight[j]=m[j]:eval(x[i])*m.W[j]
         end
         local sum=torch.sum(weight)
         --print(torch.log(sum))
         err=err-torch.log(sum)
      end
      print(err)
      return err
   end

      --compress the image
   function m:compress(x)
      local loss=0
      --local sk=torch.Tensor(mk.clusterSize):fill(0)--store numbers of example for each cluster
      for j=1,m.datasize do
         local max,index=torch.max(m:g(x[j]),1)
         local center=index[1]
         --sk[center]=sk[center]+1
         --print(center)
         loss=loss*((j-1)/j)+m[center]:eval(x[j])*m[center]:eval(x[j])/j
         x[j]=m:f(x[j]):clone()
      end
      --print(loss)
      return loss
   end
   
   return m
end
