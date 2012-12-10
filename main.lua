

--[[
Sample Main File
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 11/28/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

This file shows an example of using the tile utilities.
]]

require("image")
dofile("tile.lua")
dofile("mcluster.lua")
dofile("kmeans.lua")
dofile("mog.lua")
-- An example of using tile
function main()
   local K=torch.Tensor({arg[1]})[1]
   -- Read file
   im = tile.imread('boat.png')
   -- Convert to 7500*64 tiles representing 8x8 patches
   t = tile.imtile(im,{8,8})

   --[[
   local t_mixg=t:clone()
   local mixg=mog(64,8)
   mixg:learn(t,1e-3,1e-3)
   mixg:compress(t_mixg)
   --]]
   ----[[
   local t_kmeans=t:clone()
   local mk=kmeans(64,K)
   mk:learn(t)
   local Number_of_bits=mk:histogram(t)
   local loss=mk:compress(t_kmeans)
   print("K="..K.."  Compress_loss="..loss)
   print("K="..K.."  Number_of_bits="..Number_of_bits)
   --]]
   
   -- Convert back to 800*600 image with 8x8 patches
   im2 = tile.tileim(t_kmeans,{8,8},{600,800})
   -- Show the image
   image.display(im2)
   -- The following call can save the image
   -- tile.imwrite(im2,'boat2.png')
end

main()
