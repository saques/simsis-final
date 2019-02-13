function compareHeights(zinit)
  
  h1 = load(strcat("h-", num2str(zinit), "-sloped-0/stats.txt"))(:,1);
  h2 = load(strcat("h-", num2str(zinit), "-sloped-1/stats.txt"))(:,1);
  
  X = 1:length(h1);
  
  plot(X, h1);
  hold on;
  plot(X, h2);
  
  xlabel("Frame");
  ylabel("Altura (m)");
  
  l = legend('Theta = 0°','Theta = 45°'); 
  set(l, "fontsize", 15, "location", "northeast");
  set(gca, "fontsize", 15);
  
endfunction
