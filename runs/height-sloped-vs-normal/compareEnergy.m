function compareEnergy(file1, file2, interval, removeG = true, l1 = "Theta = 0°", l2 = "Theta = 45°", m=3, g = 9.81)
  
  h1 = load(strcat(file1, "/stats.txt"))(:,1);
  h2 = load(strcat(file2, "/stats.txt"))(:,1);
  
  e1 = load(strcat(file1, "/stats.txt"))(:,3);
  e2 = load(strcat(file2, "/stats.txt"))(:,3);
  
  if (removeG)
    e1 -= m*g*h1;
    e2 -= m*g*h2;
  endif
  
  X = interval;
  
  plot(X, e1(interval));
  hold on;
  plot(X, e2(interval));
  
  xlabel("Frame");
  if(removeG)
    ylabel("Energia cinetica (J)");
  else
    ylabel("Energia (J)");
  endif
  
  l = legend(l1,l2); 
  set(l, "fontsize", 15, "location", "northeast");
  set(gca, "fontsize", 15);
  
endfunction