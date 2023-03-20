
class CoresetTreeService(CoresetServiceBase):
    pass

    def build_from_file(self, file_path: Union[Union[str, os.PathLike], Iterable[Union[str, os.PathLike]]], target_file_path: Union[Union[str, os.PathLike], Iterable[Union[str, os.PathLike]]]=None, *, reader_f: Callable=pd.read_csv, reader_kwargs: dict=None, reader_chunk_size_param_name: str=None, chunk_size: int=None, chunk_by: Union[Callable, str, list]=None) -> 'CoresetTreeService':
        """
        Create a coreset tree based on data taken from local storage.
        build functions may be called only once. To add more data to the coreset tree use one of
        the partial_build functions. All features must be numeric.
        The target will be ignored when the Coreset is built.

        Parameters:
            file_path: file, list of files, directory, list of directories.
                Path(s) to the place where data is stored.
                Data includes features, may include targets and may include indices.

            target_file_path: file, list of files, directory, list of directories, optional.
                Use when the dataset files are split to features and target.
                Each file should include only one column.

            reader_f: pandas like read method, optional, default pandas read_csv.
                For example, to read excel files use pandas read_excel.

            reader_kwargs: dict, optional.
                Keyword arguments used when calling reader_f method.

            reader_chunk_size_param_name: str, optional.
                reader_f input parameter name for reading file in chunks.
                When not provided we'll try to figure it out our self.
                Based on the data, we decide on the optimal chunk size to read
                and use this parameter as input when calling reader_f.
                Use "ignore" to skip the automatic chunk reading logic.

            chunk_size: int, optional.
                The number of instances used when creating a coreset node in the tree.
                chunk_size=0: nodes are created based on input chunks.

            chunk_by: function, label, or list of labels, optional.
                Split the data according to the provided key.
                When provided, chunk_size input is ignored.

        Returns:
            self"""
        pass

    def partial_build_from_file(self, file_path: Union[Union[str, os.PathLike], Iterable[Union[str, os.PathLike]]], target_file_path: Union[Union[str, os.PathLike], Iterable[Union[str, os.PathLike]]]=None, *, reader_f: Callable=pd.read_csv, reader_kwargs: dict=None, reader_chunk_size_param_name: str=None, chunk_size: int=None, chunk_by: Union[Callable, str, list]=None) -> 'CoresetTreeService':
        """
        Add new samples to a coreset tree based on data taken from local storage.
        All features must be numeric. The target will be ignored when the Coreset is built.

        Parameters:
            file_path: file, list of files, directory, list of directories.
                Path(s) to the place where data is stored.
                Data includes features, may include targets and may include indices.

            target_file_path: file, list of files, directory, list of directories, optional.
                Use when files are split to features and target.
                Each file should include only one column.

            reader_f: pandas like read method, optional, default pandas read_csv.
                For example, to read excel files use pandas read_excel.

            reader_kwargs: dict, optional.
                Keyword arguments used when calling reader_f method.

            reader_chunk_size_param_name: str, optional.
                reader_f input parameter name for reading file in chunks.
                When not provided we'll try to figure it out our self.
                Based on the data, we decide on the optimal chunk size to read
                and use this parameter as input when calling reader_f.
                Use "ignore" to skip the automatic chunk reading logic.

            chunk_size: int, optional, default previous used chunk_size.
                The number of instances used when creating a coreset node in the tree.
                chunk_size=0: nodes are created based on input chunks.

            chunk_by: function, label, or list of labels, optional.
                Split the data according to the provided key.
                When provided, chunk_size input is ignored.

        Returns:
            self"""
        pass

    def build_from_df(self, datasets: Union[Iterator[pd.DataFrame], pd.DataFrame], target_datasets: Union[Iterator[pd.DataFrame], pd.DataFrame]=None, *, chunk_size: int=None, chunk_by: Union[Callable, str, list]=None, copy: bool=False) -> 'CoresetTreeService':
        """
        Create a coreset tree from pandas DataFrame(s).
        build functions may be called only once. To add more data to the coreset tree use one of
        the partial_build functions. All features must be numeric.
        The target will be ignored when the Coreset is built.

        Parameters:
            datasets: pandas DataFrame or a DataFrame iterator.
                Data includes features, may include labels and may include indices.

            target_datasets: pandas DataFrame or a DataFrame iterator, optional.
                Use when data is split to features and target.
                Should include only one column.

            chunk_size: int, optional.
                The number of instances used when creating a coreset node in the tree.
                chunk_size=0: nodes are created based on input chunks.

            chunk_by: function, label, or list of labels, optional.
                Split the data according to the provided key.
                When provided, chunk_size input is ignored.

            copy: boolean, default False.
                False (default) - Input data might be updated as result of functions such as update_targets or update_features.
                True - Data is copied before processing (impacts memory).

        Returns:
            self"""
        pass

    def partial_build_from_df(self, datasets: Union[Iterator[pd.DataFrame], pd.DataFrame], target_datasets: Union[Iterator[pd.DataFrame], pd.DataFrame]=None, *, chunk_size: int=None, chunk_by: Union[Callable, str, list]=None, copy: bool=False) -> 'CoresetTreeService':
        """
        Add new samples to a coreset tree based on the pandas DataFrame iterator.
        All features must be numeric. The target will be ignored when the Coreset is built.

        Parameters:
            datasets: pandas DataFrame or a DataFrame iterator.
                Data includes features, may include targets and may include indices.

            target_datasets: pandas DataFrame or a DataFrame iterator, optional.
                Use when data is split to features and target.
                Should include only one column.

            chunk_size: int, optional, default previous used chunk_size.
                The number of instances used when creating a coreset node in the tree.
                chunk_size=0: nodes are created based on input chunks.

            chunk_by: function, label, or list of labels, optional.
                Split the data according to the provided key.
                When provided, chunk_size input is ignored.

            copy: boolean, default False.
                False (default) - Input data might be updated as result of functions such as update_targets or update_features.
                True - Data is copied before processing (impacts memory).

        Returns:
            self"""
        pass

    def build(self, X: Union[Iterable, Iterable[Iterable]], y: Union[Iterable[Any], Iterable[Iterable[Any]]]=None, indices: Union[Iterable[Any], Iterable[Iterable[Any]]]=None, props: Union[Iterable[Any], Iterable[Iterable[Any]]]=None, *, chunk_size: int=None, chunk_by: Union[Callable, str, list]=None, copy: bool=False) -> 'CoresetTreeService':
        """
        Create a coreset tree from the parameters X, y, indices and props (properties).
        build functions may be called only once. To add more data to the coreset tree use one of
        the partial_build functions. All features must be numeric.
        The target will be ignored when the Coreset is built.

        Parameters:
            X: array like or iterator of arrays like.
                An array or an iterator of features. All features must be numeric.

            y: array like or iterator of arrays like, optional.
                An array or an iterator of targets.
                The target will be ignored when the Coreset is built.

            indices: array like or iterator of arrays like, optional.
                An array or an iterator with indices of X.

            props: array like or iterator of arrays like, optional.
                An array or an iterator of properties.
                Properties, won’t be used to compute the Coreset or train the model, but it is possible to
                filter_out_samples on them or to pass them in the select_from_function of get_important_samples.

            chunk_size: int, optional.
                The number of instances used when creating a coreset node in the tree.
                chunk_size=0: nodes are created based on input chunks.

            chunk_by: function, label, or list of labels, optional.
                Split the data according to the provided key.
                When provided, chunk_size input is ignored.

            copy: boolean, default False.
                False (default) - Input data might be updated as result of functions such as update_targets or update_features.
                True - Data is copied before processing (impacts memory).

        Returns:
            self"""
        pass

    def partial_build(self, X: Union[Iterable, Iterable[Iterable]], y: Union[Iterable[Any], Iterable[Iterable[Any]]]=None, indices: Union[Iterable[Any], Iterable[Iterable[Any]]]=None, props: Union[Iterable[Any], Iterable[Iterable[Any]]]=None, *, chunk_size: int=None, chunk_by: Union[Callable, str, list]=None, copy: bool=False) -> 'CoresetTreeService':
        """
        Add new samples to a coreset tree from parameters X, y, indices and props (properties).
        All features must be numeric. The target will be ignored when the Coreset is built.

        Parameters:
            X: array like or iterator of arrays like.
                An array or an iterator of features. All features must be numeric.

            y: array like or iterator of arrays like, optional.
                An array or an iterator of targets.
                The target will be ignored when the Coreset is built.

            indices: array like or iterator of arrays like, optional.
                An array or an iterator with indices of X.

            props: array like or iterator of arrays like, optional.
                An array or an iterator of properties.
                Properties, won’t be used to compute the Coreset or train the model, but it is possible to
                filter_out_samples on them or to pass them in the select_from_function of get_important_samples.

            chunk_size: int, optional.
                The number of instances used when creating a coreset node in the tree.
                chunk_size=0: nodes are created based on input chunks.

            chunk_by: function, label, or list of labels, optional.
                Split the data according to the provided key.
                When provided, chunk_size input is ignored.

            copy: boolean, default False
                False (default) - Input data might be updated as result of functions such as update_targets or update_features.
                True - Data is copied before processing (impacts memory).

        Returns:
            self"""
        pass

    def save(self, dir_path: Union[str, os.PathLike]=None, name: str=None, save_buffer: bool=True, override: bool=False, allow_pickle: bool=True) -> pathlib.Path:
        """
        Save service configuration and relevant data to a local directory.
        Use this method when the service needs to be restored.

        Parameters:
            dir_path: string or PathLike, optional, default self.working_directory.
                A local directory for saving service's files.

            name: string, optional, default service class name (lower case).
                Name of the subdirectory where the data will be stored.

            save_buffer: boolean, default True.
                Save also the data in the buffer (a partial node of the tree)
                along with the rest of the saved data.

            override: bool, optional, default False.
                False: add a timestamp suffix so each save won’t override the previous ones.
                True: The existing subdirectory with the provided name is overridden.

            allow_pickle: bool, optional, default True.
                True: Saves the Coreset tree in pickle format (much faster).
                False: Saves the Coreset tree in JSON format.

        Returns:
            Save directory path."""
        pass

    def load(cls, dir_path: Union[str, os.PathLike], name: str=None, *, data_manager: DataManagerT=None, load_buffer: bool=True, working_directory: Union[str, os.PathLike]=None) -> 'CoresetTreeService':
        """
        Restore a service object from a local directory.

        Parameters:
            dir_path: str, path.
                Local directory where service data is stored.

            name: string, optional, default service class name (lower case).
                The name prefix of the subdirectory to load.
                When several subdirectories having the same name prefix are found, the last one, ordered by name, is selected.
                For example when saving with override=False, the chosen subdirectory is the last saved.

            data_manager: DataManagerBase subclass, optional.
                When specified, input data manger will be used instead of restoring it from the saved configuration.

            load_buffer: boolean, optional, default True.
                If set, load saved buffer (a partial node of the tree) from disk and add it to the tree.

            working_directory: str, path, optional, default use working_directory from saved configuration.
                Local directory where intermediate data is stored.

        Returns:
            CoresetTreeService object"""
        pass

    def get_coreset(self, level: int=0, as_orig: bool=False, with_index: bool=False) -> dict:
        """
        Get tree's coreset data either in a processed format or in the original format.
        Use the level parameter to control the level of the tree from which samples will be returned.

        Parameters:
            level: int, optional, default 0.
                Defines the depth level of the tree from which the coreset is extracted.
                Level 0 returns the coreset from the head of the tree with up to coreset_size samples.
                Level 1 returns the coreset from the level below the head of the tree with up to 2*coreset_size samples, etc.

            as_orig: boolean, optional, default False.
                Should the data be returned in its original format or as a tuple of indices, X, and optionally y.
                True: data is returned as a pandas DataFrame.
                False: return a tuple of (indices, X, y) if target was used and (indices, X) when there is no target.

            with_index: boolean, optional, default False.
                Relevant only when as_orig=True. Should the returned data include the index column.

        Returns:
            Dict:
                data: numpy arrays tuple (indices, X, optional y) or a pandas DataFrame.
                w: A numpy array of sample weights.
                n_represents: number of instances represented by the coreset."""
        pass

    def save_coreset(self, file_path: Union[str, os.PathLike], level: int=0, as_orig: bool=False, with_index: bool=False):
        """
        Get coreset from the tree and save to a file along with coreset weights.
        Use the level parameter to control the level of the tree from which samples will be returned.

        Parameters:
            file_path: string or PathLike.
                Local file path to store the coreset.

            level: int, optional, default 0.
                Defines the depth level of the tree from which the coreset is extracted.
                Level 0 returns the coreset from the head of the tree with up to coreset_size samples.
                Level 1 returns the coreset from the level below the head of the tree with up to 2*coreset_size samples, etc.


            as_orig: boolean, optional, default False.
                True: save in the original format.
                False: save in a processed format (indices, X, y, weight).

            with_index: boolean, optional, default False.
                Relevant only when as_orig=True. Save also index column."""
        pass

    def get_important_samples(self, size: int=None, ignore_indices: Iterable=None, select_from_indices: Iterable=None, select_from_function: Callable[[Iterable, Iterable, Union[Iterable, None], Union[Iterable, None]], Iterable[Any]]=None, ignore_seen_samples: bool=True) -> Union[ValueError, dict]:
        '''
        Returns indices of samples in descending order of importance. Useful for identifying mislabeled instances.
        Either class_size (recommended) or size must be provided. Must be called after build.

        Parameters:
            size: int, optional.
                Number of samples to return.
                When class_size is provided, remaining samples are taken from classes not appearing in class_size dictionary.

            ignore_indices: array-like, optional.
                An array of indices to ignore when selecting important samples.

            select_from_indices: array-like, optional.
                 An array of indices to consider when selecting important samples.

            select_from_function: function, optional.
                 Pass a function in order to limit the selection of the important samples accordingly.
                 The function should accept 4 parameters as input: indices, X, y, props.
                 and return a list(iterator) of the desired indices.

            ignore_seen_samples: bool, optional, default True.
                 Exclude already seen samples and set the seen flag on any indices returned by the function.

        Returns:
            Dict:
                indices: array-like[int].
                    Important samples indices.
                X: array-like[int].
                    X array.
                y: array-like[int].
                    y array.
                importance: array-like[float].
                    The importance property. Instances that receive a high Importance in the Coreset computation,
                    require attention as they usually indicate a labeling error,
                    anomaly, out-of-distribution problem or other data-related issue.

        Examples
        -------
        Input:
            size=100,
            class_size={"class A": 10, "class B": 50, "class C": "all"}
        Output:
            10 of "class A",
            50 of "class B",
            12 of "class C" (all),
            28 of "class D/E"'''
        pass

    def set_seen_indication(self, seen_flag: bool=True, indices: Iterable=None):
        """
        Set samples as 'seen' or 'unseen'. Not providing an indices list defaults to setting the flag on all samples.

        Parameters:
            seen_flag: bool, optional, default True.
                Set 'seen' or 'unseen' flag

            indices: array like, optional.
                Set flag only for the provided list of indices. Defaults to all indices."""
        pass

    def remove_samples(self, indices: Iterable, force_resample_all: int=None, force_sensitivity_recalc: int=None, force_do_nothing: bool=False):
        """
        Remove samples from the coreset tree.
        The coreset tree is automatically updated to accommodate to the changes.

        Parameters:
            indices: array-like.
                An array of indices to be removed from the coreset tree.

            force_resample_all: int, optional.
                Force full resampling of the affected nodes in the coreset tree, starting from level=force_resample_all.
                None - Do not force_resample_all (default), 0 - The head of the tree, 1 - The level below the head of the tree,
                len(tree)-1 = leaf level, -1 - same as leaf level.

            force_sensitivity_recalc: int, optional.
                Force the recalculation of the sensitivity and partial resampling of the affected nodes,
                based on the coreset's quality, starting from level=force_sensitivity_recalc.
                None - If self.save_all=False - one level above leaf node level. If self.save_all=True - leaf level
                0 - The head of the tree, 1 - The level below the head of the tree,
                len(tree)-1 = leaf level, -1 - same as leaf level.

            force_do_nothing: bool, optional, default False.
                When set to True, suppresses any update to the coreset tree until update_dirty is called."""
        pass

    def update_targets(self, indices: Iterable, y: Iterable, force_resample_all: int=None, force_sensitivity_recalc: int=None, force_do_nothing: bool=False):
        """
        Update the targets for selected samples on the coreset tree.
        The coreset tree is automatically updated to accommodate to the changes.

        Parameters:
            indices: array-like.
                An array of indices to be updated.

            y: array-like.
                An array of classes/labels. Should have the same length as indices.

            force_resample_all: int, optional.
                Force full resampling of the affected nodes in the coreset tree, starting from level=force_resample_all.
                None - Do not force_resample_all (default), 0 - The head of the tree, 1 - The level below the head of the tree,
                len(tree)-1 = leaf level, -1 - same as leaf level.

            force_sensitivity_recalc: int, optional.
                Force the recalculation of the sensitivity and partial resampling of the affected nodes,
                based on the coreset's quality, starting from level=force_sensitivity_recalc.
                None - If self.save_all=False - one level above leaf node level. If self.save_all=True - leaf level
                0 - The head of the tree, 1 - The level below the head of the tree,
                len(tree)-1 = leaf level, -1 - same as leaf level.

            force_do_nothing: bool, optional, default False.
                When set to True, suppresses any update to the coreset tree until update_dirty is called."""
        pass

    def update_features(self, indices: Iterable, X: Iterable, feature_names: Iterable[str]=None, force_resample_all: int=None, force_sensitivity_recalc: int=None, force_do_nothing: bool=False):
        """
        Update the features for selected samples on the coreset tree.
        The coreset tree is automatically updated to accommodate to the changes.

        Parameters:
            indices: array-like.
                An array of indices to be updated.

            X: array-like.
                An array of features. Should have the same length as indices.

            feature_names:
                If the quantity of features in X is not equal to the quantity of features in the original coreset,
                this param should contain list of names of passed features.

            force_resample_all: int, optional.
                Force full resampling of the affected nodes in the coreset tree, starting from level=force_resample_all.
                None - Do not force_resample_all (default), 0 - The head of the tree, 1 - The level below the head of the tree,
                len(tree)-1 = leaf level, -1 - same as leaf level.

            force_sensitivity_recalc: int, optional.
                Force the recalculation of the sensitivity and partial resampling of the affected nodes,
                based on the coreset's quality, starting from level=force_sensitivity_recalc.
                None - If self.save_all=False - one level above leaf node level. If self.save_all=True - leaf level
                0 - The head of the tree, 1 - The level below the head of the tree,
                len(tree)-1 = leaf level, -1 - same as leaf level.

            force_do_nothing: bool, optional, default False.
                When set to True, suppresses any update to the coreset tree until update_dirty is called."""
        pass

    def filter_out_samples(self, filter_function: Callable[[Iterable, Iterable, Union[Iterable, None], Union[Iterable, None]], Iterable[Any]], force_resample_all: int=None, force_sensitivity_recalc: int=None, force_do_nothing: bool=False):
        """
        Remove samples from the coreset tree, based on the provided filter function.
        The coreset tree is automatically updated to accommodate to the changes.

        Parameters:
            filter_function: function, optional.
                A function that returns a list of indices to be removed from the tree.
                The function should accept 4 parameters as input: indices, X, y, props
                and return a list(iterator) of indices to be removed from the coreset tree.
                For example, in order to remove all instances with a target equal to 6, use the following function:
                filter_function = lambda indices, X, y, props : indices[y = 6].

            force_resample_all: int, optional.
                Force full resampling of the affected nodes in the coreset tree, starting from level=force_resample_all.
                None - Do not force_resample_all (default), 0 - The head of the tree, 1 - The level below the head of the tree,
                len(tree)-1 = leaf level, -1 - same as leaf level.

            force_sensitivity_recalc: int, optional.
                Force the recalculation of the sensitivity and partial resampling of the affected nodes,
                based on the coreset's quality, starting from level=force_sensitivity_recalc.
                None - If self.save_all=False - one level above leaf node level. If self.save_all=True - leaf level
                0 - The head of the tree, 1 - The level below the head of the tree,
                len(tree)-1 = leaf level, -1 - same as leaf level.

            force_do_nothing: bool, optional, default False.
                When set to True, suppresses any update to the coreset tree until update_dirty is called."""
        pass

    def update_dirty(self, force_resample_all: int=None, force_sensitivity_recalc: int=None):
        """
        Calculate the sensitivity and resample the nodes that were marked as dirty,
        meaning they were affected by any of the methods:
        remove_samples, update_targets, update_features or filter_out_samples,
        when they were called with force_do_nothing.

        Parameters:
            force_resample_all: int, optional.
                Force full resampling of the affected nodes in the coreset tree, starting from level=force_resample_all.
                None - Do not force_resample_all (default), 0 - The head of the tree, 1 - The level below the head of the tree,
                len(tree)-1 = leaf level, -1 - same as leaf level.

            force_sensitivity_recalc: int, optional.
                Force the recalculation of the sensitivity and partial resampling of the affected nodes,
                based on the coreset's quality, starting from level=force_sensitivity_recalc.
                None - If self.save_all=False - one level above leaf node level. If self.save_all=True - leaf level
                0 - The head of the tree, 1 - The level below the head of the tree,
                len(tree)-1 = leaf level, -1 - same as leaf level."""
        pass

    def is_dirty(self) -> bool:
        """
        Returns:
            Indicates whether the coreset tree has nodes marked as dirty, meaning they were affected by any of the methods: remove_samples, update_targets, update_features or filter_out_samples, when they were called with force_do_nothing."""
        pass

    def fit(self, level: int=0, model: Any=None, **model_params):
        """
        Fit a model on the coreset tree.
        Parameters:
            level: Defines the depth level of the tree from which the coreset is extracted.
                Level 0 returns the coreset from the head of the tree with up to coreset_size samples.
                Level 1 returns the coreset from the level below the head of the tree with up to 2*coreset_size samples, etc.

            model: An ML model instance, optional.
                When provided, model_params are not relevant.
                Default: instantiate the service model class using input model_params.

            model_params: Model hyperparameters kwargs.
                Input when instantiating default model class.

        Returns:
            Fitted estimator."""
        pass

    def predict(self, X: Union[Iterable, Iterable[Iterable]]):
        """
        Run prediction on the trained model.
        Parameters:
            X: An array of features.
        Returns:
            Model prediction results."""
        pass

    def predict_proba(self, X: Union[Iterable, Iterable[Iterable]]):
        """
        Run prediction on the trained model.
        Parameters:
            X: An array of features.
        Returns:
            Returns the probability of the sample for each class in the model."""
        pass

    def print(self):
        """
        Print the tree's string representation."""
        pass

    def plot(self, dir_path: Union[str, os.PathLike]=None, name: str=None) -> pathlib.Path:
        """
        Produce a tree graph plot and save figure as a local png file.
        Parameters:
            dir_path: string or PathLike.
                Path to save the plot figure in; if not provided, or if isn't valid/doesn't exist,
                the figure will be saved in the current directory (from which this method is called).

            name: string, optional.
                Name of the image file.
        Returns:
            Image file path"""
        pass
