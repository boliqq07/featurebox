Build Model
===========

Using data::

    >>> in_data, y = pd.read_pickle("in_data_no_sgt.pkl_pd")

    >>> y = y.astype(np.float32)
    >>> y = torch.from_numpy(y)
    >>> X_train, y_train, X_test, y_test = train_test(*in_data, y, random_state=0)
    >>> gen = GraphGenerator(*X_train, targets=y_train)
    >>> test_gen = GraphGenerator(*X_test, targets=y_test)

    >>> loader1 = MGEDataLoader(
    ...    dataset=gen,
    ...    batch_size=2000,
    ...    shuffle=False,
    ...    num_workers=0,)
    >>> loader2 = MGEDataLoader(
    ...    dataset=test_gen,
    ...    batch_size=2000,
    ...    shuffle=False,
    ...    num_workers=0,)

Net model::

    >>> model = CrystalGraphConvNet(atom_fea_len=36, nbr_fea_len=1,
    ...                            # state_fea_len=2,
    ...                            inner_atom_fea_len=64, n_conv=3, h_fea_len=(256, 128, 64), n_h=2,)


where the ``atom_fea_len``, ``nbr_fea_len``, ``state_fea_len``, should be consist with the data,
which depend on the :class:`CrystalBgGraph` and so on.

Training and test::

    >>> bl = BaseLearning(model, loader1,test_loader=loader2, device="cuda:1",
    ...                  opt=None, clf=False)
    >>> torch.save(bl.model.state_dict(), './parameter_n_sgt.pkl')
    >>> bl.run(500)
