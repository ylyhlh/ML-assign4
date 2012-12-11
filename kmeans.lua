--[[
K-Means clustering algorithm implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 11/28/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

The k-means algorithm should be presented here. You can implement it in any way
you want. For your convenience, a clustering object is provided at mcluster.lua

Here is how I implemented it:

kmeans(n,k) is a constructor to return an object km which will perform k-means
algorithm on data of dimension n with k clusters. The km object stores the i-th
cluster at km[i], which is an mcluster object. The km object has the following
methods:

km:g(x): the decision function to decide which cluster the vector x belongs to.
Return a scalar representing cluster index.

km:f(x): the output function to output a prototype that could replace vector x.

km:learn(x): learn the clusters using x, which is a m*n matrix representing m
data samples.
]]

dofile("mcluster.lua")

-- Create a k-means learner
-- n: dimension of data
-- k: number of clusters
function kmeans(n,k)
-- Remove the following line and add your stuff
--print("You have to define this function by yourself!");

   local mk={}
   mk.features=n--number of features
   mk.clusterSize=k--number of clusters
   mk.datasize=0--number of titles
   mk.center=torch.Tensor(1):fill(0)--store the center for each example
   mk.respons=torch.Tensor(1,1):fill(0)--responsibilities
   for i=1,mk.clusterSize do
      mk[i]=mcluster(n)
   end
   --km:g(x): the decision function to decide which cluster the vector x belongs to.
   function mk:g(x)
      local index=0
      local min=1E16
      for i=1,mk.clusterSize do
         if mk[i]:eval(x)<min then
            index=i
            min=mk[i]:eval(x)
         end
      end
      return index
   end
   --km:f(x): the output function to output a prototype that could replace vector x.
   function mk:f(x)
      return mk[mk:g(x)].m
   end
   --km:learn(x): learn the clusters using x, which is a m*n matrix representing m data samples.
   function mk:learn(x)
      local max_iter=100
      local datasize= x:size(1)
      --init clustering
      local perm=torch.randperm(datasize)
      for i=1,mk.clusterSize do
         mk[i]:set_m(x[perm[i]])
         --print(perm[i])
      end
      --iteration
      local sk=torch.Tensor(mk.clusterSize):fill(0)--store numbers of example for each cluster
      mk.center:resize(datasize):fill(0)--store the center for each example
      mk.respons:resize(datasize,mk.clusterSize):fill(0)
      for i=1,max_iter do
         print("This is kmean "..i.." -th step")
         sk:fill(0)
         local flag=0 --flag to determine whether converge
         --compute center for each example
         for j=1,datasize do
            local tmp=mk:g(x[j])
            if tmp~=mk.center[j] then
               flag=1
            end
            if mk.center[j]~=0 then
               mk.respons[j][mk.center[j]]=0
            end
            mk.center[j]=tmp
            sk[mk.center[j]]=sk[mk.center[j]]+1
            mk.respons[j][mk.center[j]]=1
            --print(tmp)
         end
         
         --print(torch.min(sk))
         if flag==0 then 
            --print(i)
            break
         end
         --update cluster
         for j=1,mk.clusterSize do
         --[[
            local dataset=torch.Tensor(sk[j],mk.features)
            local count=0
            for l=1,datasize do
               if mk.center[l]==j then
                  count=count+1
                  dataset[count]=x[l]:clone()
               end
            end
            mk[j]:learn(dataset,torch.ones(sk[j]))
            ]]
            if torch.sum(mk.respons:select(2,j))<=0 then
               print("one cluster converge to one point")
               mk[j]:set_m(x[torch.randperm(datasize)[1]])
            else
               --print(torch.sum(mk.respons:select(2,j)))
               mk[j]:learn(x,mk.respons:select(2,j))
            end 
         end
      end
   end
   --compress the image
   function mk:compress(x)
      local datasize= x:size(1)
      local loss=0
      --local sk=torch.Tensor(mk.clusterSize):fill(0)--store numbers of example for each cluster
      for j=1,datasize do
         local center=mk:g(x[j])
         --sk[center]=sk[center]+1
         --print(center)
         loss=loss*((j-1)/j)+mk[center]:eval(x[j])*mk[center]:eval(x[j])/j
         x[j]=mk:f(x[j]):clone()
      end
      --print(loss)
      return loss
   end
   --compute Number_of_bits
   function mk:histogram(x)
      local datasize= x:size(1)
      local Hk=torch.Tensor(mk.clusterSize):fill(0)
      for i=1,mk.clusterSize do
         Hk[i]=torch.sum(torch.eq(mk.center,torch.Tensor(datasize):fill(i)))/datasize
      end
      --print(Hk)
      local Histogram_Entropy= -torch.sum(Hk:cmul(torch.log(Hk):div(torch.log(2))))
      local Number_of_bits =datasize * Histogram_Entropy

      return Number_of_bits
   end

   return mk
end
