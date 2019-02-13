function [idx1, idx2, max1, max2, rel] = comparePeaks(file1, file2, range1, range2)
  
  h1 = load(strcat(file1, "/stats.txt"))(:,1);
  h2 = load(strcat(file2, "/stats.txt"))(:,1);
 
 
  [m1, i1] = max(h1(range1));
  [m2, i2] = max(h2(range2));
  
  max1= m1;
  max2= m2;
  rel = max1/max2;
  idx1 = i1;
  idx2 = i2;
  
endfunction
