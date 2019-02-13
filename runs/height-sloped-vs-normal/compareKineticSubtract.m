function compareKineticSubtract(file1, file2, interval, m=3, g = 9.81)
  
  h1 = load(strcat(file1, "/stats.txt"))(:,1);
  h2 = load(strcat(file2, "/stats.txt"))(:,1);
  
  e1 = load(strcat(file1, "/stats.txt"))(:,3);
  e2 = load(strcat(file2, "/stats.txt"))(:,3);
  
  e1 -= m*g*h1;
  e2 -= m*g*h2;
  
  e1 -= e2;
  
  X = interval;
  
  plot(X, e1(interval));
  
  xlabel("Frame");
  
  ylabel("Diferencias de energia cinetica (J)");

  set(gca, "fontsize", 15);
  
endfunction