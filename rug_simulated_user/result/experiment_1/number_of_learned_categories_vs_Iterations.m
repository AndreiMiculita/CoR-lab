figure ();
hold on;
grid on;
set(gca,'LineStyleOrder', '-');
NLI= [1];
plot(NLI(1:size(NLI,2)), 1:size(NLI,2), '--O', 'Color',[1 0 0], 'LineWidth',2.)
xlabel('Question / Correction Iterations','FontSize',15);
ylabel('Number of Learned Categories','FontSize',15);