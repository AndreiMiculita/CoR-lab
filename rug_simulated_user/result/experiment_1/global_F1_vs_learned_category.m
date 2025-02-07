close all;
figure ();
hold on;
grid on;
set(gca,'LineStyleOrder', '--');
globalF1= [0];
axis([0,5,0.5,1.05]);
plot(1:size(globalF1,2), globalF1(1:size(globalF1,2)),'-.O', 'Color',[0 0 1], 'LineWidth',2.);
xlabel('Number of Learned Categories','FontSize',15);
ylabel('Global Classification Accuracy','FontSize',15);